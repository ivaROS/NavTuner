#!/usr/bin/env python
from __future__ import print_function
import copy

import genpy
import numpy as np
import subprocess
import multiprocessing as mp
import os
import sys
from tqdm import tqdm
import rosbag
import rospkg
import roslaunch
import time
import test_driver
from gazebo_driver_v2 import GazeboDriver
import rosgraph
import threading
import Queue
from ctypes import c_bool
import codecs
import signal

import socket
import contextlib

from testing_scenarios import TestingScenarios
import dynamic_reconfigure.client
from actionlib_msgs.msg import GoalStatusArray
from nav_msgs.msg import Odometry
import rospy
import csv
import datetime
import pickle
from result_recorder import ResultRecorders
from rospy_message_converter.message_converter import convert_dictionary_to_ros_message as convert_dict
from std_msgs.msg import Float32
from collections import defaultdict
from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor, HuberRegressor
from rosbag.bag import ROSBagUnindexedException
from cv_bridge import CvBridge
from sensor_msgs.msg import LaserScan, Image


class RosMsgUnicodeErrors:
    def __init__(self):
        self.msg_type = None

    def __call__(self, err):
        global _warned_decoding_error
        if self.msg_type not in _warned_decoding_error:
            _warned_decoding_error.add(self.msg_type)
            # Lazy import to avoid this cost in the non-error case.
            import logging
            logger = logging.getLogger('rosout')
            extra = "message %s" % self.msg_type if self.msg_type else "unknown message"
            logger.error("Characters replaced when decoding %s (will print only once): %s", extra, err)
        return codecs.replace_errors(err)


def port_in_use(port):
    with contextlib.closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as sock:
        if sock.connect_ex(('127.0.0.1', port)) == 0:
            print("Port " + str(port) + " is in use")
            return True
        else:
            print("Port " + str(port) + " is not in use")
            return False


class GazeboMaster(mp.Process):
    def __init__(self, task_queue, result_queue, kill_flag, soft_kill_flag, ros_port, gazebo_port, gazebo_launch_mutex,
                 result_recorders, model, best_configs, ranges, path=None, suffix='classifier', **kwargs):
        super(GazeboMaster, self).__init__()
        self.daemon = False

        self.path = path if path is not None else '/home/haoxin/data/training/'

        self.task_queue = task_queue
        self.result_queue = result_queue
        self.ros_port = ros_port
        self.gazebo_port = gazebo_port
        self.gazebo_launch_mutex = gazebo_launch_mutex
        self.result_recorders = result_recorders
        self.models = model
        self.best_configs = best_configs
        self.ranges = ranges
        self.suffix = suffix

        self.core = None
        self.gazebo_launch = None
        self.controller_launch = None
        self.gazebo_driver = None
        self.current_world = None
        self.kill_flag = kill_flag
        self.soft_kill_flag = soft_kill_flag
        self.is_shutdown = False
        self.had_error = False

        self.gui = True

        print("New master")

        self.ros_master_uri = "http://localhost:" + str(self.ros_port)
        self.gazebo_master_uri = "http://localhost:" + str(self.gazebo_port)
        os.environ["ROS_MASTER_URI"] = self.ros_master_uri
        os.environ["GAZEBO_MASTER_URI"] = self.gazebo_master_uri

        if self.gui == False:
            if 'DISPLAY' in os.environ:
                del os.environ['DISPLAY']  # To ensure that no GUI elements of gazebo activated
        else:
            if 'DISPLAY' not in os.environ:
                os.environ['DISPLAY'] = ':0'

    def run(self):
        while not self.is_shutdown and not self.had_error:
            self.process_tasks()
            time.sleep(5)
            if not self.is_shutdown:
                print("(Not) Relaunching on " + str(
                    os.getpid()) + ", ROS_MASTER_URI=" + self.ros_master_uri, file=sys.stderr)
        print("Run totally done")

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
                folder = self.path + task["robot"] + '_' + task["controller"] + '_' + str(task['min_obstacle_spacing']) + '/'
                filename = folder
                if not os.path.isdir(folder):
                    os.makedirs(folder)
                for key in task['params'].keys():
                    filename += str(key) + str(task['params'][key]) + '_'
                filename += 'seed' + str(task['seed'])
                recorders = ResultRecorders(filename)
                recorder_list = []
                for recorder in self.result_recorders:
                    r = recorders.get_recorders(recorder)
                    recorder_list.append(r)
                    r.start()
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
                            # planner = rospy.get_param("/move_base/dynamic_reconfigure")
                            # dr_controller = dynamic_reconfigure.client.Client(planner, timeout=30)
                            # dr_controller.update_configuration(task['params'])
                            # if 'max_global_plan_lookahead_dist' in task['params'].keys():
                            #     look_ahead = 2. * task['params']['max_global_plan_lookahead_dist'] + 1.
                            #     dr_costmap = dynamic_reconfigure.client.Client('/move_base/local_costmap', timeout=30)
                            #     dr_costmap.update_configuration({'height': look_ahead, 'width': look_ahead,
                            #                                      'resolution': 0.01 * look_ahead})
                            print("Running test...")

                            # master = rosgraph.Master('/mynode')

                            # TODO: make this a more informative type
                            result = test_driver.run_test(goal_pose=scenario.getGoal())

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
                        recorder.write(task['params'])
                    elif recorder.key is 'laser_scan' or (recorder.key is "depth"):
                        recorder.close()
                recorders.close()
                print("result saved!")
                self.return_result(task)

                if self.had_error:
                    print(result, file=sys.stderr)


            except Queue.Empty as e:
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

    def start_core(self):

        # env_prefix = "ROS_MASTER_URI="+ros_master_uri + " GAZEBO_MASTER_URI=" + gazebo_master_uri + " "

        my_command = "roscore -p " + str(self.ros_port)

        # my_env = os.environ.copy()
        # my_env["ROS_MASTER_URI"] = self.ros_master_uri
        # my_env["GAZEBO_MASTER_URI"] = self.gazebo_master_uri

        print("Starting core...")
        self.core = subprocess.Popen(my_command.split())  # preexec_fn=os.setsid
        print("Core started! [" + str(self.core.pid) + "]")

    def roslaunch_core(self):

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        # roslaunch.configure_logging(uuid)

        self.core = roslaunch.parent.ROSLaunchParent(
            run_id=uuid, roslaunch_files=[],
            is_core=True, port=self.ros_port
        )
        self.core.start()

    def roslaunch_controller(self, robot, controller_name, controller_file=None, controller_args=None):

        # controller_path =

        if controller_args is None:
            controller_args = {}
        if controller_file is None:
            rospack = rospkg.RosPack()
            path = rospack.get_path("nav_scripts")
            launch_files = [path + "/launch/" + robot + "_" + controller_name + "_controller.launch"]
            # We'll assume Gazebo is launched are ready to go
        else:
            launch_files = controller_file

        uuid = roslaunch.rlutil.get_or_generate_uuid(None, True)
        # roslaunch.configure_logging(uuid)
        # print path

        # Remapping stdout to /dev/null
        sys.stdout = open(os.devnull, "w")

        for key, value in controller_args.items():
            var_name = "GM_PARAM_" + key.upper()
            value = str(value)
            os.environ[var_name] = value
            print("Setting environment variable [" + var_name + "] to '" + value + "'")

        self.controller_launch = roslaunch.parent.ROSLaunchParent(
            run_id=uuid, roslaunch_files=launch_files,
            is_core=False, port=self.ros_port  # , roslaunch_strs=controller_args
        )
        self.controller_launch.start()

        sys.stdout = sys.__stdout__

    def roslaunch_gazebo(self, world):
        if world == self.current_world:
            if not self.gazebo_launch._shutting_down:
                return
            else:
                print("Gazebo crashed, restarting")

        if self.gazebo_launch is not None:
            self.gazebo_launch.shutdown()

        self.current_world = world

        # This will wait for a roscore if necessary, so as long as we detect any failures
        # in start_roscore, we should be fine
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, True)
        # roslaunch.configure_logging(uuid) #What does this do?
        # print path

        # Without the mutex, we frequently encounter this problem:
        # https://bitbucket.org/osrf/gazebo/issues/821/apparent-transport-race-condition-on
        with self.gazebo_launch_mutex:
            self.gazebo_launch = roslaunch.parent.ROSLaunchParent(
                run_id=uuid, roslaunch_files=[world],
                is_core=False, port=self.ros_port
            )
            self.gazebo_launch.start()

        try:
            msg = rospy.wait_for_message("/odom", Odometry, 30)
        except rospy.exceptions.ROSException:
            print("Error! odometry not received!")
            return False

        return True

    def shutdown(self):
        self.is_shutdown = True

    # TODO: add conditional logic to trigger this
    def task_error(self, task):
        self.task_queue.put(task)
        self.task_queue.task_done()
        self.shutdown()

    def return_result(self, result):
        resultcopy = copy.deepcopy(result)
        # resultcopy.pop("laser_scan")
        print("Returning completed task: " + str(resultcopy))
        self.result_queue.put(result)
        self.task_queue.task_done()


class GazeboTester(GazeboMaster):
    def __init__(self, task_queue, result_queue, kill_flag, soft_kill_flag, ros_port, gazebo_port, gazebo_launch_mutex,
                 result_recorders, model, best_configs, ranges, path=None, suffix='classifier', **kwargs):
        super(GazeboTester, self).__init__(task_queue, result_queue, kill_flag, soft_kill_flag, ros_port, gazebo_port, gazebo_launch_mutex,
                 result_recorders, model, best_configs, ranges, path, suffix, **kwargs)

        self.path = path if path is not None else '/home/haoxin/data/test/'

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
                            print("Running test...")

                            # master = rosgraph.Master('/mynode')

                            # TODO: make this a more informative type
                            result = test_driver.run_test_predict(goal_pose=scenario.getGoal(), models=self.models,
                                                                  predict_recorder=predict_recorder, ranges=self.ranges,
                                                                  density=scenario.density, truth=self.best_configs,
                                                                  suffix=self.suffix)

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
                    elif recorder.key is 'laser_scan' or (recorder.key is "depth"):
                        recorder.close()
                recorders.close()
                print("result saved!")
                # if 'accuracy' in task:
                #     print("accuracy: {}".format(task['accuracy']))
                self.return_result(task)

                if self.had_error:
                    print(result, file=sys.stderr)


            except Queue.Empty as e:
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


class MultiMasterCoordinator:
    def __init__(self, result_recorders=None, gazebo=None, model=None, best_configs=None, ranges=None, path=None,
                 suffix=''):
        if result_recorders is None:
            result_recorders = []
        signal.signal(signal.SIGINT, self.signal_shutdown)
        signal.signal(signal.SIGTERM, self.signal_shutdown)
        self.children_shutdown = mp.Value(c_bool, False)
        self.soft_shutdown = mp.Value(c_bool, False)

        self.should_shutdown = False

        self.num_masters = 1

        self.save_results = True
        self.result_recorders = result_recorders
        self.model = model
        self.best_configs = best_configs
        self.ranges = ranges
        self.path = path
        self.suffix = suffix

        self.task_queue_capacity = 2000  # 2*self.num_masters
        self.task_queue = mp.JoinableQueue(maxsize=self.task_queue_capacity)
        self.result_queue_capacity = 2000  # *self.num_masters
        self.result_queue = mp.JoinableQueue(maxsize=self.result_queue_capacity)
        self.gazebo_masters = []
        self.gazebo = GazeboMaster if gazebo is None else gazebo
        self.result_list = []
        self.gazebo_launch_mutex = mp.Lock()

        self.fieldnames = ["controller"]
        self.fieldnames.extend(TestingScenarios.getFieldNames())
        self.fieldnames.extend(["pid", "result", "time", "path_length", "robot"])
        self.fieldnames.extend(["sim_time", "obstacle_cost_mode", "sum_scores"])

    def start(self):
        self.startResultsProcessing()
        self.startProcesses()

    def startResultsProcessing(self):
        self.result_thread = threading.Thread(target=self.processResults, args=[self.result_queue])
        self.result_thread.daemon = True
        self.result_thread.start()

    def startProcesses(self):
        self.ros_port = 11311
        self.gazebo_port = self.ros_port + 100
        for ind in xrange(self.num_masters):
            self.addProcess()

    def addProcess(self):
        while port_in_use(self.ros_port):
            self.ros_port += 1

        while port_in_use(self.gazebo_port):
            self.gazebo_port += 1

        gazebo_master = self.gazebo(self.task_queue, self.result_queue, self.children_shutdown, self.soft_shutdown,
                                    self.ros_port, self.gazebo_port, gazebo_launch_mutex=self.gazebo_launch_mutex,
                                    result_recorders=self.result_recorders, model=self.model, ranges=self.ranges,
                                    best_configs=self.best_configs, path=self.path, suffix=self.suffix)
        gazebo_master.start()
        self.gazebo_masters.append(gazebo_master)

        self.ros_port += 1
        self.gazebo_port += 1

        time.sleep(1)

    def processResults(self, queue):
        '''
        # outputfile_name = "~/Documents/dl3_gazebo_results_" + str(datetime.datetime.now())
        outputfile_name = "/data/fall2018/chapter_experiments/chapter_experiments_" + str(datetime.datetime.now())
        outputfile_name = os.path.expanduser(outputfile_name)

        with open(outputfile_name, 'wb') as csvfile:
            seen = set()
            fieldnames = [x for x in self.fieldnames if not (x in seen or seen.add(
                x))]  # http://www.martinbroadhurst.com/removing-duplicates-from-a-list-while-preserving-order-in-python.html

            datawriter = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='', extrasaction='ignore')
            datawriter.writeheader()

            while not self.should_shutdown:  # This means that results stop getting saved to file as soon as I try to kill it
                try:
                    task = queue.get(block=False)

                    result_string = "Result of ["
                    for k, v in task.iteritems():
                        # if "result" not in k:
                        result_string += str(k) + ":" + str(v) + ","
                    result_string += "]"

                    print result_string

                    if "error" not in task:
                        self.result_list.append(result_string)
                        if self.save_results:
                            datawriter.writerow(task)
                            csvfile.flush()
                    else:
                        del task["error"]
                        self.task_queue.put(task)
                        self.addProcess()

                    # print "Result of " + task["world"] + ":" + task["controller"] + "= " + str(task["result"])
                    queue.task_done()
                except Queue.Empty, e:
                    # print "No results!"
                    time.sleep(1)
                    '''
        while not self.should_shutdown:  # This means that results stop getting saved to file as soon as I try to kill it
            try:
                task = queue.get(block=False)
                result_string = "Result of ["
                for k, v in task.iteritems():
                    # if "result" not in k:
                    result_string += str(k) + ":" + str(v) + ","
                result_string += "]"

                # print result_string
                if "error" not in task:
                    self.result_list.append(result_string)
                    # if self.save_results:
                    #     datawriter.writerow(task)
                    #     csvfile.flush()
                    # pickle.dump(data_dict, open(filename, 'w'))
                    print("result saved!")
                else:
                    del task["error"]
                    self.task_queue.put(task)
                    self.addProcess()

                # print "Result of " + task["world"] + ":" + task["controller"] + "= " + str(task["result"])
                queue.task_done()
            except Queue.Empty as e:
                # print "No results!"
                time.sleep(1)

    def signal_shutdown(self, signum, frame):
        self.shutdown()

    def shutdown(self):
        with self.children_shutdown.get_lock():
            self.children_shutdown.value = True

        for process in mp.active_children():
            process.join()

        self.should_shutdown = True

        # for process in self.gazebo_masters:
        #    if process.is_alive():
        #        process.join()
        # sys.exit(0)

    def waitToFinish(self):
        print("Waiting until everything done!")
        self.task_queue.join()
        print("All tasks processed!")
        with self.soft_shutdown.get_lock():
            self.soft_shutdown.value = True

        # The problem is that this won't happen if I end prematurely...
        self.result_queue.join()
        print("All results processed!")

        for result in self.result_list:
            print(result)

    # This list should be elsewhere, possibly in the configs package
    def addTasks(self, tasks):
        for task in tasks:
            self.task_queue.put(task)

    # This list should be elsewhere, possibly in the configs package


def generate_training_data(tasks, path=None):
    start_time = time.time()
    result_recorders = [{"topic": "result"}, {"topic": "time"}, {"topic": "path_length"},
                        {"topic": "laser_scan", "node": "/scan"}, {"topic": "params"}]  # {"topic": "depth", "node": "/camera/depth/image_raw"}
    master = MultiMasterCoordinator(result_recorders=result_recorders, path=path)
    master.start()
    master.addTasks(tasks)
    master.waitToFinish()
    master.shutdown()
    end_time = time.time()
    print("Total time: " + str(end_time - start_time))


def find_all_success(file):
    seed = list(range(0, 100, 4))
    for f in file:
        try:
            bag = rosbag.Bag(f, 'r')
            for _, result, _ in bag.read_messages(topics=['result']):
                if result.__class__.__name__ != '_std_msgs__String' or result.data != 'SUCCEEDED':
                    s = int(f[:-4].split('seed')[1])
                    if s in seed:
                        seed.remove(s)
        except ROSBagUnindexedException:
            continue

    return seed


def find_best_config(dir, params, path):
    file = [dir + f for f in os.listdir(dir) if os.path.isfile(dir + f)]
    best_config = {}
    seed = find_all_success(file)
    for p in params.keys():
        range_list = list(params[p]['range'])
        l = len(range_list)
        score = np.zeros(l)
        success = np.zeros(l)
        pl = np.zeros(l)
        t = np.zeros(l)
        num_pl = 0
        num_t = 0
        for i in range(l):
            value = range_list[i]
            file_pv = [f for f in file if p + str(value) in f]
            for f in file_pv:
                s = int(f[:-4].split('seed')[1])
                try:
                    bag = rosbag.Bag(f, 'r')
                    for _, result, _ in bag.read_messages(topics=['result']):
                        if result.data == 'SUCCEEDED':
                            success[i] += 1
                            if s not in seed:
                                break
                            for _, path_length, _ in bag.read_messages(topics=['path_length']):
                                pl[i] += path_length.data
                                num_pl += 1
                            for _, rt, _ in bag.read_messages(topics=['time']):
                                t[i] += rt.data.secs
                                num_t += 1
                    bag.close()
                except ROSBagUnindexedException:
                    # print(f)
                    continue
            pl[i] /= num_pl
            t[i] /= num_t
        print('success: {}, path length: {}, time: {}'.format(success, pl, t))
        # s_rank = find_rank(success)
        # pl_rank = find_rank(pl)
        # t_rank = find_rank(t)

        score = normalize(success)-normalize(pl) - normalize(t)
        print(score)
        best = np.argmax(score)
        best_config[p] = best  # range_list[best]
    print(best_config)
    return best_config


def normalize(x):
    return (x - np.mean(x)) / np.std(x)


def find_rank(array):
    ind = array.argsort()
    rank = np.empty_like(ind)
    rank[ind] = np.arange(len(array))

    return rank


def read_data_from_bag(files, topics, interval=1, counter=0):
    # data = defaultdict(lambda: [])
    data = None
    bridge = CvBridge()
    for f in tqdm(files):
        try:
            bag = rosbag.Bag(f, 'r')
            D = None
            for t in topics:
                d = None
                # counter = 0
                while True:
                    try:
                        for _, msg, _ in bag.read_messages(topics=[t]):
                            if counter % interval == 0:
                                if t is 'laser_scan':
                                    msg_data = np.array(msg.ranges)
                                    mask = np.isnan(msg_data)
                                    if np.sum(mask) == len(mask):
                                        # print('invalid data')
                                        counter = max(0, counter-1)
                                        continue
                                    msg_data[mask] = msg.range_max + 1
                                elif t is 'depth':
                                    msg.width = 320
                                    msg.height = 240
                                    msg_data = np.array(bridge.imgmsg_to_cv2(msg, "passthrough"), dtype=np.float32)
                                if d is None:
                                    d = np.reshape(msg_data, newshape=(1,) + np.shape(np.squeeze(msg_data)))
                                else:
                                    msg_data = np.reshape(msg_data, newshape=(1,) + np.shape(np.squeeze(msg_data)))
                                    d = np.concatenate([d, msg_data], axis=0)
                            counter += 1
                        break
                    except (genpy.message.DeserializationError, rosbag.bag.ROSBagFormatException) as e:
                        print("{} error {}".format(f, e))
                        # counter = np.maximum(counter-1, 0)
                        # counter += 1
                        break
            if D is None:
                D = d
            else:
                D = np.concatenate([D, d], axis=0)
        except ROSBagUnindexedException as e:
            print("{} error {}".format(f, e))

        if data is None:
            data = D
        else:
            if D is not None:
                data = np.concatenate([data, D], axis=0)
        bag.close()

    return data


def find_max(interval=1):
    # data = defaultdict(lambda: [])
    topics = ['laser_scan']
    dir = '/home/haoxin/data/training/lookahead/turtlebot_teb_49/'
    files = [dir + f for f in os.listdir(dir) if os.path.isfile(dir + f)]
    data = 0.
    for f in files:
        try:
            bag = rosbag.Bag(f, 'r')
            for t in topics:
                counter = 0
                while True:
                    try:
                        for _, msg, _ in bag.read_messages(topics=[t]):
                            if counter % interval == 0:
                                if t is 'laser_scan':
                                    msg_data = np.array(msg.ranges)
                                    mask = np.isnan(msg_data)
                                    if np.sum(mask) == len(mask):
                                        # print('invalid data')
                                        # counter = max(0, counter-1)
                                        continue
                                    msg_data[mask] = 0.
                                # print(np.amax(msg_data))
                                    data = np.maximum(data, np.amax(msg_data))
                        break
                    except (genpy.message.DeserializationError, rosbag.bag.ROSBagFormatException) as e:
                        print("{} error {}".format(f, e))
                        # counter = np.maximum(counter-1, 0)
                        # counter += 1
                        break
        except ROSBagUnindexedException as e:
            print("{} error {}".format(f, e))
        bag.close()
    print(data)
    return data


def find_results(dir, params, seed=None):
    file = [dir + f for f in os.listdir(dir) if os.path.isfile(dir + f)]
    results = {}
    for p in params.keys():
        # range_list = list(params[p]['range'])
        # l = len(range_list)
        # score = np.zeros(l)
        success = 0.  # np.zeros(l)
        pl = 0.  # np.zeros(l)
        t = 0.  # np.zeros(l)
        num_s = 0
        num_pl = 0
        num_t = 0
        results[p] = {}
        # for i in range(l):
        #     value = range_list[i]
        # file_pv = [f for f in file if p + str(value) in f]
        for f in file:
            s = int(f.split('seed')[-1][:-4])
            if seed is not None and s not in seed:
                continue
            try:
                bag = rosbag.Bag(f, 'r')
                for _, result, _ in bag.read_messages(topics=['result']):
                    num_s += 1
                    if result.data in ['SUCCEEDED', 'BUMPER_COLLISION']: #
                        if result.data == 'SUCCEEDED':
                            success += 1.
                        for _, path_length, _ in bag.read_messages(topics=['path_length']):
                            pl += path_length.data
                            # print(f.split('/')[-1], path_length.data)
                            num_pl += 1
                        for _, rt, _ in bag.read_messages(topics=['time']):
                            t += rt.data.secs
                            num_t += 1
                bag.close()
            except ROSBagUnindexedException:
                continue
        # success /= num_s
        pl /= max(num_pl, 1)
        t /= max(num_t, 1)
        results[p] = {'success rate': success, 'path_length': np.round(pl,2), 'runtime': t}

    return results


def test(read_path, params, ranges, save_path, suffix='gt', scene=None, id=0):
    start_time = time.time()
    # path = '/home/haoxin/data/training/'
    dir = [read_path + d for d in os.listdir(read_path) if os.path.isdir(read_path + d)]
    density = [float(d.split('_')[-1]) for d in dir]
    # best_configs = {}
    training_data = {}
    models = {}
    filename = read_path + 'best_configs.pickle'
    best_configs = pickle.load(open(filename, 'rb'))
    '''
    for p in params:
        training_data[p] = {'X': None, 'Y': None, 'd': None}
    training_topics = ['laser_scan']
    
    # pickle.dump(best_configs, open(filename, 'wb'))
    for i in range(len(dir)):
        d = dir[i] + '/'
        # best_configs[density[i]] = find_best_config(d, params, read_path)
        files = [d + f for f in os.listdir(d) if os.path.isfile(d + f)]
        X = read_data_from_bag(files, training_topics, interval=5, counter=i)
        # print(np.shape(X))
        ones = np.ones(np.shape(X)[0], dtype=np.int)
        for p in params:
            best_config = best_configs[density[i]]
            if training_data[p]['X'] is None:
                training_data[p]['X'] = X
                # training_data[p]['d'] = i * ones
                if suffix == 'regressor':
                    training_data[p]['Y'] = ranges[p][best_config[p]] * ones
                else:
                    training_data[p]['Y'] = best_config[p] * ones.astype(np.int)
            else:
                training_data[p]['X'] = np.concatenate([training_data[p]['X'], X], axis=0)
                # training_data[p]['d'] = np.concatenate([training_data[p]['d'], i * ones], axis=0)
                if suffix == 'regressor':
                    training_data[p]['Y'] = np.concatenate([training_data[p]['Y'], ranges[p][best_config[p]] * ones],
                                                           axis=0)
                else:
                    training_data[p]['Y'] = np.concatenate([training_data[p]['Y'], best_config[p] * ones.astype(np.int)]
                                                           , axis=0)
    # '''
    # '''
    # print(training_data['planner_frequency']['X'].shape)
    # filename = read_path + suffix + '_alldata.pickle'
    # pickle.dump(training_data, open(filename, 'wb'))
    # training_data = pickle.load(open(filename, 'rb'))
    # '''
    filename = read_path + suffix + '_models.pickle'
    '''
    for p in params:
        if suffix == 'regressor':
            models[p] = SGDRegressor(loss='huber').fit(training_data[p]['X'], training_data[p]['Y'])
        else:
            models[p] = SGDClassifier(loss='log').fit(training_data[p]['X'], training_data[p]['Y'])
    pickle.dump(models, open(filename, 'wb'))
    # '''
    # '''
    models = pickle.load(open(filename, 'rb'))
    end_time = time.time()
    print("Total time: " + str(end_time - start_time))
    start_time = time.time()
    result_recorders = [{"topic": "result"}, {"topic": "time"}, {"topic": "path_length"},
                        {"topic": "predict"}, {"topic": "accuracy"}]
    master = MultiMasterCoordinator(result_recorders=result_recorders, gazebo=GazeboTester, model=models, ranges=ranges,
                                    best_configs=best_configs, path=save_path + 'model/' + suffix + '/', suffix=suffix)
    master.start()
    tasks = []
    no = id * 50 if scene in ['sector', 'maze', 'empty'] else 125 * id
    scene = 'maze_predict' if scene is None else scene + '_predict'
    ss = 100
    se = 125
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
    # '''
    end_time = time.time()
    print("Total time: " + str(end_time - start_time))


def test_gt(read_path, params, ranges, save_path, scene=None, id=0):
    test(read_path, params, ranges, save_path, 'gt', scene=scene, id=id)


def test_classifier(read_path, params, ranges, save_path, scene=None, id=0):
    test(read_path, params, ranges, save_path, 'classifier', scene=scene, id=id)


def test_regressor(read_path, params, ranges, save_path, scene=None, id=0):
    test(read_path, params, ranges, save_path, 'regressor', scene=scene, id=id)


def test_default(save_path, scene=None, id=0):
    # '''
    start_time = time.time()
    result_recorders = [{"topic": "result"}, {"topic": "time"}, {"topic": "path_length"}]
    master = MultiMasterCoordinator(result_recorders=result_recorders, gazebo=GazeboMaster, path=save_path + 'default/')
    master.start()
    tasks = []
    no = id * 50 if scene in ['sector', 'maze', 'empty'] else 125 * id
    ss = 100
    se = 125
    scene = 'maze_predict' if scene is None else scene + '_predict'
    for scenario in [scene]:
        for controller in ['ego_teb']:
            # 'ego_teb', 'p2d', 'p2d_local_global', 'laser_classifier_weighted', 'laser_classifier_local_global', 'intention'
            for seed in range(ss, se):
                task = {'scenario': scenario, 'controller': controller, 'seed': seed,
                        'params': {'max_depth': 3.},
                        'robot': 'turtlebot', 'min_obstacle_spacing': 1.25,
                        'num_obstacles': no, 'maze_file': 'maze_1.25.pickle'}
                tasks.append(task)
    master.addTasks(tasks)
    master.waitToFinish()
    master.shutdown()
    end_time = time.time()
    print("Total time: " + str(end_time - start_time))
    # '''
    # params = {'max_global_plan_lookahead_dist': 3.}


def print_results(save_path, suffix=['classifier', 'regressor']):
    for controller in ['intention_local_global']: #'p2d', 'p2d_local_global', 'laser_classifier_weighted', 'laser_classifier_local_global',
                                        #'intention', 'ego_teb'
        print(controller)
        task = {'scenario': 'empty_predict', 'controller': controller, 'seed': 0,
                'params': {'max_depth': 3.},
                'robot': 'turtlebot', 'min_obstacle_spacing': 1.25, 'num_obstacles': 200,
                'controller_args': {'sim_time': 2}}
        default_result = find_results(save_path + 'default/' + task["robot"] + '_' + task["controller"] +
                                      '_' + str(task["min_obstacle_spacing"]) + '/', task['params'])
        print('default:')
        print(default_result)
        # gt_result = find_results(save_path + '/model/gt/' + task["robot"] + '_' + task["controller"] + '/', task['params'])
        # print('gt:')
        # print(gt_result)
        for s in suffix:
            model_result = find_results(save_path + '/model/' + s + '/' + task["robot"] + '_' + task["controller"] + '/',
                                        task['params'])
            print(s+':')
            print(model_result)


def generate_data():
    tasks = []
    values = np.linspace(1., 5.5, 10)
    # values = [0.0625, 0.125, 0.25, 0.5, 1.0]  #
    # '''
    densities = [0.75, 1.0, 1.25, 1.5]
    for scenario in ['maze']:
        for controller in ['ego_teb']:
            for seed in range(0, 50):
                for min_dist in densities:
                    for lookahead in values:
                        params = {'max_depth': lookahead}
                        task = {'scenario': scenario, 'controller': controller, 'seed': seed,
                                'robot': 'turtlebot', 'num_obstacles': 200, 'min_obstacle_spacing': min_dist,
                                'params': params, 'maze_file': 'maze_1.25.pickle', 'use_maze': True}
                        tasks.append(task)
    # '''
    generate_training_data(tasks, path='data/training/max_depth/')


def predict():
    freq_values = [0.0625, 0.125, 0.25, 0.5, 1.0]#
    depth_values = np.linspace(1., 5.5, 10)
    p = 'max_depth'
    read_path = 'data/training/' + p + '/'
    folder = {0: '/' + p + '_0/', 1: '/'+p+'_50/', 2: '/'+p+'_100/', 3: '/'+p+'_150/', 4: '/'+p+'/'}
    ranges = {'max_depth': depth_values,}  # 'planner_frequency': freq_values,
    params = {'max_depth': {'range': depth_values},}  # 'planner_frequency': {'range': freq_values},
    for scene in ['campus', 'fourth_floor', 'sector']:  #'maze', 'campus', 'fourth_floor', 'sector'
        print('\n' + scene)
        for id in range(5):
            save_path = 'data/test/' + scene + folder[id] #'/max_depth_0/'
            test_gt(read_path, params, ranges, save_path, scene=scene)
            test_regressor(read_path, params, ranges, save_path, scene=scene, id=id)
            test_classifier(read_path, params, ranges, save_path, scene=scene, id=id)
            test_default(save_path, scene=scene, id=id)
            print_results(save_path, suffix=[])  # 'classifier', 'regressor'


if __name__ == "__main__":
    # generate_data()
    predict()
