from __future__ import print_function
from future import standard_library
standard_library.install_aliases()
from builtins import str
from builtins import range
import queue
import multiprocessing as mp
import os
import pickle
import rospy
import sys
import time

import numpy as np
import rosbag
import torch
from rosbag import ROSBagUnindexedException
from rospy_message_converter.message_converter import convert_dictionary_to_ros_message as convert_dict

import test_driver
from gazebo_master import GazeboTester, MultiMasterCoordinator, find_results
from result_recorder import ResultRecorders
from testing_scenarios import TestingScenarios
from DQN import DQNAgent, CNNDQNAgent, MultiDQNAgent


class GazeboRL(GazeboTester):
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
                    if r.key == 'predict':
                        predict_recorder = r
                    if r.key == 'reward':
                        reward_recorder = r
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
                            print("Running test...")

                            # master = rosgraph.Master('/mynode')

                            # TODO: make this a more informative type
                            aux = task['aux'] if 'aux' in task else False
                            double = task['double'] if 'double' in task else False
                            multiple = task['multiple'] if 'multiple' in task else False
                            if multiple:
                                result = test_driver.run_test_rl_multi(goal_pose=scenario.getGoal(), models=self.models,
                                                                       predict_recorder=predict_recorder,
                                                                       reward_recorder=reward_recorder,
                                                                       ranges=self.ranges,
                                                                       shortest_path=scenario.path,
                                                                       param=['max_depth',
                                                                              'planner_frequency',
                                                                              'selection_cost_hysteresis',
                                                                              'switching_blocking_period',
                                                                              'selection_prefer_initial_plan',
                                                                              'inflation_dist',
                                                                              'feasibility_check_no_poses',
                                                                              ])
                            elif aux:
                                result = test_driver.run_test_rl_aux(goal_pose=scenario.getGoal(), models=self.models,
                                                                     predict_recorder=predict_recorder,
                                                                     reward_recorder=reward_recorder,
                                                                     ranges=self.ranges,
                                                                     param='planner_frequency',
                                                                     shortest_path=scenario.path,
                                                                     density=scenario.density)
                            elif double:
                                result = test_driver.run_test_rl_double(goal_pose=scenario.getGoal(),
                                                                        models=self.models,
                                                                        predict_recorder=predict_recorder,
                                                                        reward_recorder=reward_recorder,
                                                                        ranges=self.ranges,
                                                                        param=['max_depth', 'planner_frequency'],
                                                                        shortest_path=scenario.path,
                                                                        density=scenario.density)
                            else:
                                result = test_driver.run_test_rl(goal_pose=scenario.getGoal(), models=self.models,
                                                                 predict_recorder=predict_recorder,
                                                                 reward_recorder=reward_recorder, ranges=self.ranges,
                                                                 param='planner_frequency', shortest_path=scenario.path)
                            filepath = self.path
                            if not os.path.isdir(filepath):
                                os.makedirs(filepath)
                            if task['seed'] % 50 == 0:
                                filename = filepath + task['model_name'] + 's' + str(self.models.state_space) \
                                           + 'seed' + str(task['seed']) + '.pt'
                                self.models.save_model(filename)

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
                    if recorder.key == 'result':
                        recorder.write(task['result'])
                    elif recorder.key == 'time':
                        recorder.write(task['time'])
                    elif recorder.key == 'path_length':
                        recorder.write(task['path_length'])
                    elif recorder.key == 'params':
                        recorder.write(convert_dict('std_msgs/Float32', task['params']))
                    elif recorder.key == 'laser_scan':
                        recorder.close()
                recorders.close()
                print("result saved!")
                # if 'accuracy' in task:
                #     print("accuracy: {}".format(task['accuracy']))
                self.return_result(task)

                if self.had_error:
                    print(result, file=sys.stderr)


            except queue.Empty as e:
                with self.soft_kill_flag.get_lock():
                    if self.soft_kill_flag.value:
                        self.shutdown()
                        print("Soft shutdown requested")
                time.sleep(1)

            with self.kill_flag.get_lock():
                if self.kill_flag.value:
                    self.shutdown()

        print("Done with processing, killing launch files...")
        # It seems like killing the core should kill all of the nodes,
        # but it doesn't
        if self.gazebo_launch is not None:
            self.gazebo_launch.shutdown()

        if self.controller_launch is not None:
            self.controller_launch.shutdown()

        print("GazeboMaster shutdown: killing core...")
        self.core.shutdown()
        # self.core.kill()
        # os.killpg(os.getpgid(self.core.pid), signal.SIGTERM)
        print("All cleaned up")


class GazeboRLBC(GazeboTester):
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
                    if r.key == 'predict':
                        predict_recorder = r
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
                            print("Running test...")

                            # master = rosgraph.Master('/mynode')

                            # TODO: make this a more informative type
                            aux = task['aux'] if 'aux' in task else False
                            double = task['double'] if 'double' in task else False
                            if aux:
                                result = test_driver.run_test_rl_bc_aux(goal_pose=scenario.getGoal(),
                                                                        models=self.models,
                                                                        predict_recorder=predict_recorder,
                                                                        ranges=self.ranges, truth=self.best_configs,
                                                                        param='planner_frequency',
                                                                        density=scenario.density)
                            elif double:
                                result = test_driver.run_test_rl_bc_double(goal_pose=scenario.getGoal(),
                                                                           models=self.models,
                                                                           predict_recorder=predict_recorder,
                                                                           ranges=self.ranges, truth=self.best_configs,
                                                                           param=['max_depth', 'planner_frequency'],
                                                                           density=scenario.density)
                            else:
                                result = test_driver.run_test_rl_bc(goal_pose=scenario.getGoal(), models=self.models,
                                                                    predict_recorder=predict_recorder,
                                                                    ranges=self.ranges, truth=self.best_configs,
                                                                    param='max_depth', density=scenario.density)
                            filepath = self.path
                            if not os.path.isdir(filepath):
                                os.makedirs(filepath)
                            if task['seed'] % 100 == 0:
                                filename = filepath + task['model_name'] + 's' + str(self.models.state_space) \
                                           + 'a' + str(self.models.action_space) + 'seed' + str(task['seed']) + '.pt'
                                self.models.save_model(filename)

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
                    if recorder.key == 'result':
                        recorder.write(task['result'])
                    elif recorder.key == 'time':
                        recorder.write(task['time'])
                    elif recorder.key == 'path_length':
                        recorder.write(task['path_length'])
                    elif recorder.key == 'params':
                        recorder.write(convert_dict('std_msgs/Float32', task['params']))
                    elif recorder.key == 'laser_scan':
                        recorder.close()
                recorders.close()
                print("result saved!")
                # if 'accuracy' in task:
                #     print("accuracy: {}".format(task['accuracy']))
                self.return_result(task)

                if self.had_error:
                    print(result, file=sys.stderr)


            except queue.Empty as e:
                with self.soft_kill_flag.get_lock():
                    if self.soft_kill_flag.value:
                        self.shutdown()
                        print("Soft shutdown requested")
                time.sleep(1)

            with self.kill_flag.get_lock():
                if self.kill_flag.value:
                    self.shutdown()

        print("Done with processing, killing launch files...")
        # It seems like killing the core should kill all of the nodes,
        # but it doesn't
        if self.gazebo_launch is not None:
            self.gazebo_launch.shutdown()

        if self.controller_launch is not None:
            self.controller_launch.shutdown()

        print("GazeboMaster shutdown: killing core...")
        self.core.shutdown()
        # self.core.kill()
        # os.killpg(os.getpgid(self.core.pid), signal.SIGTERM)
        print("All cleaned up")


class GazeboRLPredict(GazeboTester):
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
                    if r.key == 'predict':
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
                            print("Running test...")

                            # master = rosgraph.Master('/mynode')

                            # TODO: make this a more informative type
                            double = task['double'] if 'double' in task else False
                            multiple = task['multiple'] if 'multiple' in task else False
                            if multiple:
                                result = test_driver.run_test_rl_predict_multi(goal_pose=scenario.getGoal(),
                                                                               models=self.models,
                                                                               predict_recorder=predict_recorder,
                                                                               ranges=self.ranges,
                                                                               param=['max_depth',
                                                                                      'planner_frequency',
                                                                                      'selection_cost_hysteresis',
                                                                                      'switching_blocking_period',
                                                                                      'selection_prefer_initial_plan',
                                                                                      'inflation_dist',
                                                                                      'feasibility_check_no_poses',
                                                                                      ])
                            elif double:
                                result = test_driver.run_test_rl_predict_double(goal_pose=scenario.getGoal(),
                                                                                models=self.models,
                                                                                predict_recorder=predict_recorder,
                                                                                ranges=self.ranges,
                                                                                param=['max_depth',
                                                                                       'planner_frequency'])
                            else:
                                result = test_driver.run_test_rl_predict(goal_pose=scenario.getGoal(),
                                                                         models=self.models,
                                                                         predict_recorder=predict_recorder,
                                                                         ranges=self.ranges,
                                                                         param='planner_frequency')

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
                    if recorder.key == 'result':
                        recorder.write(task['result'])
                    elif recorder.key == 'time':
                        recorder.write(task['time'])
                    elif recorder.key == 'path_length':
                        recorder.write(task['path_length'])
                    elif recorder.key == 'params':
                        recorder.write(convert_dict('std_msgs/Float32', task['params']))
                    elif recorder.key == 'laser_scan':
                        recorder.close()
                recorders.close()
                print("result saved!")
                # if 'accuracy' in task:
                #     print("accuracy: {}".format(task['accuracy']))
                self.return_result(task)

                if self.had_error:
                    print(result, file=sys.stderr)


            except queue.Empty as e:
                with self.soft_kill_flag.get_lock():
                    if self.soft_kill_flag.value:
                        self.shutdown()
                        print("Soft shutdown requested")
                time.sleep(1)

            with self.kill_flag.get_lock():
                if self.kill_flag.value:
                    self.shutdown()

        print("Done with processing, killing launch files...")
        # It seems like killing the core should kill all of the nodes,
        # but it doesn't
        if self.gazebo_launch is not None:
            self.gazebo_launch.shutdown()

        if self.controller_launch is not None:
            self.controller_launch.shutdown()

        print("GazeboMaster shutdown: killing core...")
        self.core.shutdown()
        # self.core.kill()
        # os.killpg(os.getpgid(self.core.pid), signal.SIGTERM)
        print("All cleaned up")


def warm_start(read_path, save_path, suffix, checkpoint=None, aux=False, double=False):
    filename = read_path + 'best_configs.pickle'
    best_configs = pickle.load(open(filename, 'rb'))
    result_recorders = [{"topic": "result"}, {"topic": "time"}, {"topic": "path_length"},
                        {"topic": "predict"}, {"topic": "reward"}]
    values = [0.0625, 0.125, 0.25, 0.5, 1.0]  #
    depth_values = np.linspace(1., 5.5, 10)
    ranges = {'planner_frequency': values, 'max_depth': depth_values}
    state_space = 640
    action_space = len(depth_values)
    model = DQNAgent(state_space, action_space, aux=aux, double=double)
    # model.cuda()
    torch.set_num_threads(1)
    '''
    if checkpoint is not None:
        model.load_model(checkpoint)
    master = MultiMasterCoordinator(result_recorders=result_recorders, gazebo=GazeboRLBC, ranges=ranges,
                                    model=model, path=save_path + 'model/' + suffix + '/', suffix=suffix,
                                    best_configs=best_configs)
    master.start()
    start_time = time.time()
    tasks = []
    s0 = 1000
    se = 1200
    for scenario in ['maze_rl']:
        for controller in ['ego_teb']:
            for seed in range(s0, se):
                task = {'scenario': scenario, 'controller': controller, 'seed': seed+1,
                        'robot': 'turtlebot', 'min_obstacle_spacing': 1.25,
                        'model_name': 'bc_dqn', 'maze_file': 'maze_1.25.pickle', 'use_maze': True,
                        'num_obstacles': 200, 'double': double}
                tasks.append(task)
    master.addTasks(tasks)
    master.waitToFinish()
    master.shutdown()
    end_time = time.time()
    print "Total time: " + str(end_time - start_time)
    model.reset()
    model.train()
    # '''
    # del bc_model
    # env_model = DQNAgent(state_space, action_space)
    checkpoint = save_path + 'model/' + suffix + '/' + 'bc_dqns640a10seed1700.pt'
    # env_model.load_model(checkpoint)
    # checkpoint = torch.load(checkpoint)
    # env_model.model.load_state_dict(checkpoint['model_state_dict'])
    # model.model.load_state_dict(checkpoint['model_state_dict'])
    model.load_model(checkpoint)
    master = MultiMasterCoordinator(result_recorders=result_recorders, gazebo=GazeboRL, ranges=ranges,
                                    model=model, path=save_path + 'model/dqn/', suffix=suffix,
                                    best_configs=best_configs)
    master.start()
    start_time = time.time()
    tasks = []
    s0 = 1200
    se = 2000
    for scenario in ['maze_rl']:
        for controller in ['ego_teb']:
            for seed in range(s0, se):
                task = {'scenario': scenario, 'controller': controller, 'seed': seed + 1,
                        'robot': 'turtlebot', 'min_obstacle_spacing': 1.25,
                        'model_name': 'bc_dqn', 'maze_file': 'maze_1.25.pickle', 'use_maze': True,
                        'num_obstacles': 200, 'aux': aux, 'double': double}
                tasks.append(task)
    master.addTasks(tasks)
    master.waitToFinish()
    master.shutdown()
    end_time = time.time()
    print("Total time: " + str(end_time - start_time))


def train(save_path, suffix, checkpoint=None, aux=False, double=False):
    result_recorders = [{"topic": "result"}, {"topic": "time"}, {"topic": "path_length"},
                        {"topic": "predict"}, {"topic": "reward"}]
    values = [0.0625, 0.125, 0.25, 0.5, 1.0]  #
    depth_values = np.linspace(1., 5.5, 10)
    cost_values = [0.175, 0.35, 0.7, 1.4, 2.0]
    block_values = [0.0, 1.0, 2.0, 4.0, 8.0]
    prefer_values = [0., 0.25, .5, 0.75, 0.95]
    inflation_values = [0.1, 0.2, 0.4, 0.8, 1.6]
    poses_values = [1, 2, 5, 10, 20]
    ranges = {'planner_frequency': values, 'max_depth': depth_values, 'selection_cost_hysteresis': cost_values,
              'switching_blocking_period': block_values, 'selection_prefer_initial_plan': prefer_values,
              'inflation_dist': inflation_values, 'feasibility_check_no_poses': poses_values, }
    state_space = 640
    action_space = len(depth_values)
    model = DQNAgent(state_space, action_space, aux=aux, double=double)
    # model = MultiDQNAgent(state_space, eps=.2, eps_decay=.9993)
    # model.cuda()
    torch.set_num_threads(1)
    if checkpoint is not None:
        model.load_model(checkpoint)
    master = MultiMasterCoordinator(result_recorders=result_recorders, gazebo=GazeboRL, ranges=ranges,
                                    model=model, path=save_path + 'model/' + suffix + '/', suffix=suffix)
    master.start()
    start_time = time.time()
    tasks = []
    s0 = 11000
    se = 15000
    for scenario in ['maze_rl']:
        for controller in ['ego_teb']:
            for seed in range(s0, se):
                # if seed == 395 or seed == 565 or seed == 245 or seed == 315:
                #     continue
                task = {'scenario': scenario, 'controller': controller, 'seed': seed + 1,
                        'robot': 'turtlebot', 'min_obstacle_spacing': 1.25,
                        'model_name': 'dqn', 'maze_file': 'maze_1.25.pickle', 'use_maze': True,
                        'num_obstacles': 200, 'aux': aux, 'double': double, 'multiple': False}
                tasks.append(task)
    master.addTasks(tasks)
    master.waitToFinish()
    master.shutdown()
    # filename = save_path + 'model/' + suffix + '/' + 'dqn' + 's' + str(state_space) \
    #            + 'a' + str(action_space) + 'seed' + str(s0) + '_' + str(se) + '.pt'
    # model.save_model(filename)
    end_time = time.time()
    print("Total time: " + str(end_time - start_time))


def not_a_test(read_path, save_path, suffix='dqn', scene=None, double=False, id=0):
    result_recorders = [{"topic": "result"}, {"topic": "time"}, {"topic": "path_length"},
                        {"topic": "predict"}]
    # values = [0.0625, 0.125, 0.25, 0.5, 1.0]  #
    # depth_values = np.linspace(1., 5.5, 10)
    # ranges = {'planner_frequency': values, }  # 'max_depth': depth_values}
    values = [0.0625, 0.125, 0.25, 0.5, 1.0]  #
    depth_values = np.linspace(1., 5.5, 10)
    cost_values = [0.175, 0.35, 0.7, 1.4, 2.0]
    block_values = [0.0, 1.0, 2.0, 4.0, 8.0]
    prefer_values = [0., 0.25, .5, 0.75, 0.95]
    inflation_values = [0.1, 0.2, 0.4, 0.8, 1.6]
    poses_values = [1, 2, 5, 10, 20]
    ranges = {'planner_frequency': values, 'max_depth': depth_values, 'selection_cost_hysteresis': cost_values,
              'switching_blocking_period': block_values, 'selection_prefer_initial_plan': prefer_values,
              'inflation_dist': inflation_values, 'feasibility_check_no_poses': poses_values, }
    state_space = 640
    action_space = 10
    model = DQNAgent(state_space, action_space, eps=0., aux=False, double=double)
    # model.load_model(read_path)
    # model = MultiDQNAgent(state_space, eps=0.)
    checkpoint = torch.load(read_path)
    model.model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    start_time = time.time()
    tasks = []
    s0 = 100
    se = 150
    no = id * 50 if scene in ['sector', 'maze', 'empty'] else 125 * id
    scene = 'maze_rl' if scene is None else scene + '_predict'
    path = save_path + 'test/' + suffix + '/'
    for scenario in [scene]:
        for controller in ['ego_teb']:
            for seed in range(s0, se):
                task = {'scenario': scenario, 'controller': controller, 'seed': seed,
                        'robot': 'turtlebot', 'min_obstacle_spacing': 1.25,
                        'params': {'planner_frequncy': 1.0}, 'use_maze': True,
                        'num_obstacles': no, 'maze_file': 'maze_1.25.pickle',
                        'double': double, 'multiple': False}
    torch.set_num_threads(1)
    master = MultiMasterCoordinator(result_recorders=result_recorders, gazebo=GazeboRLPredict, ranges=ranges,
                                    model=model, path=path, suffix=suffix)
    master.start()
    master.addTasks(tasks)
    master.waitToFinish()
    master.shutdown()
    end_time = time.time()
    print("Total time: " + str(end_time - start_time))


def print_result(save_path, suffix=['dqn']):
    path = save_path + 'test/'
    task = {'scenario': 'maze_rl', 'controller': 'ego_teb', 'seed': 0,
            'robot': 'turtlebot', 'min_obstacle_spacing': 1.25,
            'params': {'max_depth': 3.0},
            'num_obstacles': 200, 'maze_file': 'maze_1.25.pickle'}
    for s in suffix:
        result = find_results(path + s + '/' + task["robot"] + '_' + task["controller"] + '/', task['params'])
        print(s + ':')
        print(result)


if __name__ == '__main__':
    # '''
    save_path = '~/simulation_data/torch/rl/max_depth/'
    checkpoint = 'data/rl/max_depth/model/dqn/dqns640seed15000.pt'
    # train(save_path, suffix='dqn', checkpoint=checkpoint, aux=False, double=True)
    train(save_path, suffix='dqn', checkpoint=None, aux=False, double=True)

    exit()
    # read_path = 'data/training/depth_planner_freq/'
    # warm_start(read_path, save_path, suffix='dqn', aux=False, double=True)
    # '''
    # '''
    p = 'max_depth'
    read_path = '/data/rl/' + p + '/model/dqn/'
    # '''
    folder = {0: '/' + p + '_0/', 1: '/'+p+'_50/', 2: '/'+p+'_100/', 3: '/'+p+'_150/', 4: '/'+p+'/'}
    for scene in ['maze', 'campus', 'fourth_floor', 'sector']:  # 
        print('\n' + scene)
        for seed in [12000, 13000, 14000, 15000]:
            for id in range(0, 5):
                print(id)
                save_path = 'data/rl' + folder[id] + scene + '/' + str(seed)
                checkpoint = read_path + 'dqns640seed'+str(seed)+'.pt'
                not_a_test(checkpoint, save_path, suffix='dqn', scene=scene, double=True, id=id)
                print_result(save_path, ['dqn', 'bc_dqn'])
    # '''
