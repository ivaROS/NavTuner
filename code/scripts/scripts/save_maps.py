from __future__ import print_function
import copy
import rospy
import subprocess
import time
import multiprocessing as mp
from nav_msgs.msg import Odometry

import roslaunch
from gazebo_utils.gazebo_world_to_map import MapCreator
from gazebo_master import GazeboMaster, MultiMasterCoordinator, port_in_use
import sys
import os
import Queue
from testing_scenarios import TestingScenarios


class GazeboMapSaver(GazeboMaster):
    def __init__(self, task_queue, result_queue, kill_flag, soft_kill_flag, ros_port, gazebo_port, gazebo_launch_mutex,
                 path=None, **kwargs):
        super(GazeboMapSaver, self).__init__(task_queue, result_queue, kill_flag, soft_kill_flag, ros_port, gazebo_port, gazebo_launch_mutex,
                 result_recorders=None, model=None, best_configs=None, ranges=None, path=path, suffix=None, **kwargs)
        self.path = path if path is not None else '/home/haoxin/data/maps/'

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
                folder = self.path
                filename = folder
                if not os.path.isdir(folder):
                    os.mkdir(folder)
                filename += 'seed' + str(task['seed'])
                scenario = scenarios.getScenario(task)

                if scenario is not None:

                    self.roslaunch_gazebo(scenario.getGazeboLaunchFile(task["robot"]))  # pass in world info
                    if not self.gazebo_launch._shutting_down:
                        try:
                            scenario.setupScenario()
                            map = subprocess.Popen(['rosrun', 'gazebo_utils', 'map_transform_publisher'])
                            print('publisher running')
                            saver = subprocess.Popen(['rosrun', 'map_server', 'map_saver', '-f', filename])
                            time.sleep(5)
                            map.terminate()
                            saver.terminate()
                        except rospy.ROSException as e:
                            task["error"] = True
                            self.had_error = True

                    else:
                        task["error"] = True
                        self.had_error = True


                task["pid"] = os.getpid()

                if self.had_error:
                    print(file=sys.stderr)


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

        print("GazeboMaster shutdown: killing core...")
        self.core.shutdown()
        # self.core.kill()
        # os.killpg(os.getpgid(self.core.pid), signal.SIGTERM)
        print("All cleaned up")


class MapCoordinator(MultiMasterCoordinator):
    def addProcess(self):
        while port_in_use(self.ros_port):
            self.ros_port += 1

        while port_in_use(self.gazebo_port):
            self.gazebo_port += 1

        gazebo_master = GazeboMapSaver(self.task_queue, self.result_queue, self.children_shutdown, self.soft_shutdown,
                                    self.ros_port, self.gazebo_port, gazebo_launch_mutex=self.gazebo_launch_mutex,
                                    path=self.path)
        gazebo_master.start()
        self.gazebo_masters.append(gazebo_master)

        self.ros_port += 1
        self.gazebo_port += 1

        time.sleep(1)


if __name__ == '__main__':
    path = '/home/haoxin/data/maps/'
    tasks = []
    for scenario in ['maze_map']:
        for controller in ['ego_teb']:
            for seed in range(200, 300):
                task = {'scenario': scenario, 'controller': controller, 'seed': seed,
                        'params': {'max_depth': 3.},
                        'robot': 'turtlebot', 'min_obstacle_spacing': 1.25, 'num_obstacles': 200}
                tasks.append(task)
    master = MapCoordinator(path=path)
    master.start()
    master.addTasks(tasks)
    master.waitToFinish()
    master.shutdown()

