import Queue
import os
import pickle
import sys
import time

import rospy

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import test_driver
from gazebo_master import MultiMasterCoordinator, GazeboTester, convert_dict, find_results
from result_recorder import ResultRecorders
from testing_scenarios import TestingScenarios
from deep_networks import Classifier, Regressor, CNNClassifier, CNNRegressor, LaserScanDataset, LaserScanDatasetBranch, \
    CNNRegressorBranch, RegressorBranch, CNNClassifierBranch, ClassifierBranch
from tqdm import tqdm


class GazeboDL(GazeboTester):
    def process_tasks(self):
        self.roslaunch_core()
        rospy.set_param('/use_sim_time', 'True')
        rospy.init_node('test_driver', anonymous=True)
        rospy.on_shutdown(self.shutdown)
        # planner = "/move_base"

        scenarios = TestingScenarios()

        self.had_error = False

        while not self.is_shutdown and not self.had_error:
            # TODO: If fail to run task, put task back on task queue
            try:
                task = self.task_queue.get(block=False)
                folder = self.path + task["robot"] + '_' + task["controller"]
                filename = folder
                if not os.path.isdir(folder):
                    os.makedirs(folder)
                filename += '/seed' + str(task['seed'])
                recorders = ResultRecorders(filename)
                recorder_list = []
                predict_recorder = None
                for recorder in self.result_recorders:
                    r = recorders.get_recorders(recorder)
                    recorder_list.append(r)
                    r.start()
                    if r.key is 'predict':
                        predict_recorder = r
                    # recorders.get_recorders(recorder)
                scenario = scenarios.getScenario(task)

                if scenario is not None:

                    self.roslaunch_gazebo(scenario.getGazeboLaunchFile(task["robot"]))  # pass in world info
                    # time.sleep(30)

                    if not self.gazebo_launch._shutting_down:

                        controller_args = task["controller_args"] if "controller_args" in task else {}

                        try:

                            scenario.setupScenario()
                            self.roslaunch_controller(task["robot"], task["controller"],
                                                      controller_args=controller_args)
                            task.update(
                                controller_args)  # Adding controller arguments to main task dict for easy logging
                            print "Running test..."

                            # master = rosgraph.Master('/mynode')

                            # TODO: make this a more informative type
                            result = test_driver.run_test_dl_branch(goal_pose=scenario.getGoal(), models=self.models,
                                                             predict_recorder=predict_recorder,
                                                             ranges=self.ranges, suffix=self.suffix,)

                        except rospy.ROSException as e:
                            result = "ROSException: " + str(e)
                            task["error"] = True
                            self.had_error = True

                        self.controller_launch.shutdown()

                    else:
                        result = "gazebo_crash"
                        task["error"] = True
                        self.had_error = True

                else:
                    result = "bad_task"

                if isinstance(result, dict):
                    task.update(result)
                else:
                    task["result"] = result
                task["pid"] = os.getpid()
                for recorder in recorder_list:
                    if recorder.key is 'result':
                        recorder.write(task['result'])
                    elif recorder.key is 'time':
                        recorder.write(task['time'])
                    elif recorder.key is 'path_length':
                        recorder.write(task['path_length'])
                    elif recorder.key is 'params':
                        recorder.write(convert_dict('std_msgs/Float32', task['params']))
                    elif recorder.key is 'laser_scan':
                        recorder.close()
                recorders.close()
                print("result saved!")
                # if 'accuracy' in task:
                #     print("accuracy: {}".format(task['accuracy']))
                self.return_result(task)

                if self.had_error:
                    print >> sys.stderr, result


            except Queue.Empty, e:
                with self.soft_kill_flag.get_lock():
                    if self.soft_kill_flag.value:
                        self.shutdown()
                        print "Soft shutdown requested"
                time.sleep(1)

            with self.kill_flag.get_lock():
                if self.kill_flag.value:
                    self.shutdown()

        print "Done with processing, killing launch files..."
        # It seems like killing the core should kill all of the nodes,
        # but it doesn't
        if self.gazebo_launch is not None:
            self.gazebo_launch.shutdown()

        if self.controller_launch is not None:
            self.controller_launch.shutdown()

        print "GazeboMaster shutdown: killing core..."
        self.core.shutdown()
        # self.core.kill()
        # os.killpg(os.getpgid(self.core.pid), signal.SIGTERM)
        print "All cleaned up"


def train(epoch, model, dataloader, loss_fn, optimizer, scheduler=None, epochs=0):
    train_loss = 0
    train_examples = 0
    model.train()
    for x, y1, y2 in tqdm(dataloader):
        train_examples += x.shape[0]
        optimizer.zero_grad()
        output1, output2 = model(x)
        if loss_fn.__class__.__name__ == 'MSELoss':
            y1 = y1.to(output1.dtype)
            y2 = y2.to(output2.dtype)
        loss = loss_fn(output1.squeeze(), y1.squeeze()) + loss_fn(output2.squeeze(), y2.squeeze())
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

    avg_train_loss = train_loss / train_examples
    scheduler.step(avg_train_loss)
    print("Epoch: {}/{}\tAvg Train Loss: {:.4f}".format(epoch, epochs, avg_train_loss))


def test(read_path, ranges, save_path, suffix='gt', cnn=False, checkpoint=None, env=None, pickleload=None, id=0):
    start_time = time.time()
    # filename = read_path + 'best_configs.pickle'
    # best_configs = pickle.load(open(filename, 'rb'))
    model_name = 'deep_'
    torch.set_num_threads(1)
    if cnn:
        model_name += 'cnn_'
    model_name += suffix
    if os.path.isfile(pickleload):
        models = {}
        checkpoints = pickle.load(open(pickleload, 'rb'))
        for param in checkpoints.keys():
            ckpt = checkpoints[param]
            if suffix == 'regressor':
                if cnn:
                    model = CNNRegressorBranch(3, 128)
                else:
                    model = RegressorBranch(256)
            else:
                if cnn:
                    model = CNNClassifierBranch(3, 128, ckpt['num_labels1'], ckpt['num_labels2'])
                else:
                    model = ClassifierBranch(256, ckpt['num_labels1'], ckpt['num_labels2'])
            model.load_state_dict(ckpt['model_state_dict'])
            model.eval()
            models[param] = model
    else:
        filename = read_path + suffix + '_alldata.pickle'
        training_data = pickle.load(open(filename, 'rb'))
        filename = read_path
        filename += model_name + '_models.pt'
        # '''
        param = 'combined'  # 'max_depth'planner_frequency
        dataset = LaserScanDatasetBranch(training_data[param]['X'], training_data[param]['Y1'], training_data[param]['Y2'])
        dataloader = DataLoader(dataset, batch_size=4096)
        if suffix == 'regressor':
            if cnn:
                model = CNNRegressorBranch(3, 128)
            else:
                model = RegressorBranch(256)
            loss_fn = nn.MSELoss()
        else:
            if cnn:
                model = CNNClassifierBranch(3, 128, 10, 5)
            else:
                model = ClassifierBranch(256, 10, 5)
            loss_fn = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5)
        if os.path.isfile(checkpoint):
            checkpoint = torch.load(checkpoint)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        EPOCHS = 0
        for epoch in range(EPOCHS):
            start_time = time.time()
            train(epoch, model, dataloader, loss_fn, optimizer, scheduler, EPOCHS)
            print('training takes {} seconds'.format(time.time() - start_time))
            if epoch % 3 == 0:
                save_dict = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'hidden_size': model.hidden_size, }
                if hasattr(model, 'kernel_size'):
                    save_dict['kernel_size'] = model.kernel_size
                if hasattr(model, 'num_labels1'):
                    save_dict['num_labels1'] = model.num_labels1
                if hasattr(model, 'num_labels2'):
                    save_dict['num_labels2'] = model.num_labels2
                torch.save(save_dict, filename)
        # pickle.dump(model, open(filename, 'wb'))
        # '''
        # models = pickle.load(open(filename, 'rb'))
        end_time = time.time()
        print "Total time: " + str(end_time - start_time)
        model.eval()
        models = {param: model}
    # '''
    start_time = time.time()
    result_recorders = [{"topic": "result"}, {"topic": "time"}, {"topic": "path_length"},
                        {"topic": "predict"}, {"topic": "accuracy"}]
    master = MultiMasterCoordinator(result_recorders=result_recorders, gazebo=GazeboDL, model=models, ranges=ranges,
                                    best_configs=None, path=save_path + 'model/' + model_name + '/',
                                    suffix=suffix)
    master.start()
    tasks = []
    no = id * 50 if env in ['sector', 'maze', 'empty'] else 125 * id
    scene = 'maze_predict' if env is None else env + '_predict'
    ss = 100
    se = 150
    for scenario in [scene]:
        for controller in ['ego_teb']:
            for seed in range(ss, se):
                task = {'scenario': scenario, 'controller': controller, 'seed': seed,
                        'robot': 'turtlebot', 'min_obstacle_spacing': 1.25,
                        'num_obstacles': no, 'maze_file': 'maze_1.25.pickle'}
                tasks.append(task)
    master.addTasks(tasks)
    master.waitToFinish()
    master.shutdown()
    end_time = time.time()
    print "Total time: " + str(end_time - start_time)
    # '''


def test_cnn_classifier(read_path, ranges, save_path, env=None, id=0):
    checkpoint = read_path + 'deep_cnn_classifier_models.pt'
    pickload = read_path + 'deep_cnn_classifier.pickle'
    test(read_path, ranges, save_path, suffix='classifier', cnn=True, checkpoint=checkpoint, env=env,
         pickleload=pickload, id=id)


def test_classifier(read_path, ranges, save_path, env=None, id=0):
    checkpoint = read_path + 'deep_classifier_models.pt'
    pickload = read_path + 'deep_classifier.pickle'
    test(read_path, ranges, save_path, suffix='classifier', cnn=False, checkpoint=checkpoint, env=env,
         pickleload=pickload, id=id)


def test_cnn_regressor(read_path, ranges, save_path, env=None, id=0):
    checkpoint = read_path + 'deep_cnn_regressor_models.pt'
    pickload = read_path + 'deep_cnn_regressor.pickle'
    test(read_path, ranges, save_path, suffix='regressor', cnn=True, checkpoint=checkpoint, env=env,
         pickleload=pickload, id=id)


def test_regressor(read_path, ranges, save_path, env=None, id=0):
    checkpoint = read_path + 'deep_regressor_models.pt'
    pickload = read_path + 'deep_regressor.pickle'
    test(read_path, ranges, save_path, suffix='regressor', cnn=False, checkpoint=checkpoint, env=env,
         pickleload=pickload, id=id)


def print_results(save_path, suffix=['classifier', 'regressor']):
    task = {'scenario': 'empty_predict', 'controller': 'ego_teb', 'seed': 0,
            'params': {'max_depth': 3.},
            'robot': 'turtlebot', 'min_obstacle_spacing': 1.25, 'num_obstacles': 200,
            'controller_args': {'sim_time': 2}}
    # gt_result = find_results(save_path + '/model/gt/' + task["robot"] + '_' + task["controller"] + '/', task['params'])
    # print('gt:')
    # print(gt_result)
    default_result = find_results(save_path + 'default/' + task["robot"] + '_' + task["controller"] +
                                  '_' + str(task["min_obstacle_spacing"]) + '/', task['params'])
    print('default:')
    print(default_result)
    for s in suffix:
        model_result = find_results(save_path + '/model/' + s + '/' + task["robot"] + '_' + task["controller"] + '/',
                                    task['params'])
        print(s + ':')
        print(model_result)
        model_result = find_results(
            save_path + '/model/deep_' + s + '/' + task["robot"] + '_' + task["controller"] + '/',
            task['params'])
        print('deep_' + s + ':')
        print(model_result)
        model_result = find_results(
            save_path + '/model/deep_cnn_' + s + '/' + task["robot"] + '_' + task["controller"] + '/',
            task['params'])
        print('deep_cnn_' + s + ':')
        print(model_result)


def predict():
    values = [0.0625, 0.125, 0.25, 0.5, 1.0]  #
    depth_values = np.linspace(1., 5.5, 10)
    p = 'combined'
    read_path = 'data/training/' + p + '/'
    folder = {0: '/' + p + '_0/', 1: '/' + p + '_50/', 2: '/' + p + '_100/', 3: '/' + p + '_150/', 4: '/' + p + '/'}
    for env in ['maze', 'campus', 'fourth_floor', 'sector']: #
        ranges = {'planner_frequency': values, 'max_depth': depth_values,}
        print('\n' + env)
        for id in range(5):
            print(id)
            save_path = '/home/haoxin/data/test/' + env + folder[id]
            test_cnn_classifier(read_path, ranges, save_path, env=env, id=id)
            test_classifier(read_path, ranges, save_path, env=env, id=id)
            test_cnn_regressor(read_path, ranges, save_path, env=env, id=id)
            test_regressor(read_path, ranges, save_path, env=env, id=id)
            print_results(save_path, suffix=['classifier', 'regressor'])


if __name__ == '__main__':
    predict()
