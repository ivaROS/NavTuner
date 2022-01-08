#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from builtins import object
from past.utils import old_div
import rospy
import random
import sys, os, time
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point, Quaternion
from sensor_msgs.msg import Image, CameraInfo, LaserScan
from copy import deepcopy
# from depth_learning.msg import GazeboState
from tf2_ros import TransformListener, Buffer, LookupException, ConnectivityException, ExtrapolationException

# from pips_test import gazebo_driver
from dynamic_reconfigure.client import Client
from gazebo_msgs.msg import ModelStates, ModelState
from gazebo_msgs.srv import SetModelState
import numpy as np
from std_srvs.srv import Empty

from gazebo_ros import gazebo_interface
import std_srvs.srv
import cv2
import pickle


# Copied from pips_test: gazebo_driver.py
# Load model xml from file
def load_model_xml(filename):
    if os.path.exists(filename):
        if os.path.isdir(filename):
            print("Error: file name is a path?", filename)
            sys.exit(0)

        if not os.path.isfile(filename):
            print("Error: unable to open file", filename)
            sys.exit(0)
    else:
        print("Error: file does not exist", filename)
        sys.exit(0)

    f = open(filename, 'r')
    model_xml = f.read()
    if model_xml == "":
        print("Error: file is empty", filename)
        sys.exit(0)

    return model_xml


class Prototype(object):
    # Copied from pips_test: gazebo_driver.py
    def barrel_points(self, xmin, ymin, xmax, ymax, min_dist, num_barrels, max_tries=100):
        '''
        # Get a dense grid of points
        points = np.mgrid[xmin:xmax:grid_size, ymin:ymax:grid_size]
        points = points.swapaxes(0, 2)
        points = points.reshape(points.size / 2, 2)
        # Choose random indexes
        idx = self.random.sample(range(points.shape[0]), num_barrels)
        print idx

        # Generate offsets
        off = self.nprandom.rand(num_barrels, 2) * grid_size / 2.0

        # Compute barrel points
        barrels = points[idx] + off
        '''

        depth = xmax - xmin
        width = ymax - ymin

        length = max(depth, width)

        n = 0
        i = 0
        barrels = []
        while n < num_barrels and i < max_tries:
            a = self.random.random()
            b = self.random.random()

            if a * length < depth and b * length < width:
                x = xmin + a * length
                y = ymin + b * length
                point = np.array((x, y))

                barrel_valid = True
                for barrel in barrels:
                    if np.linalg.norm(point - barrel) < min_dist:
                        barrel_valid = False
                        break

                if barrel_valid:
                    barrels.append(point)
                    n += 1
            i += 1

        for barrel in barrels:
            yield barrel

    def statesCallback(self, data):  # This comes in at ~100hz
        self.models = data

    def camInfoCallback(self, data):
        self.camInfo = data

    # def depthCallback(self, data):
    #     if self.haveTransform():
    #         if self.models and self.camInfo:
    #             self.saveState(data, self.models, self.camInfo, self.transform)
    #             # cv2.imshow("temp",0)
    #             # cv2.waitKey(0)
    #             self.newScene()

    def ecCallback(self, data):
        if self.haveTransform():
            if self.models and self.camInfo:
                self.saveData(data)
                # self.seed += 1
                self.newScene()

    def saveData(self, data):
        fov = 0.4
        threshold = 0.75
        num_angles = 51
        angles = np.linspace(-fov, fov, num_angles)
        idx = np.around(old_div(angles, data.angle_increment)).astype(np.int)
        msg = np.array(data.ranges)
        idx += len(msg)//2 - idx[len(idx)//2]
        labels = np.zeros((num_angles, 2))
        mask = msg[idx] >= threshold*6
        labels[mask, 0] = 1.
        labels[~mask, 1] = 1.
        self.egocircle.append(msg[np.newaxis, ...])
        self.labels.append(labels[np.newaxis, ...])
        if len(self.egocircle)%1000==1:
            print(len(self.egocircle)+1)
        if len(self.egocircle) == 10**6 + 1000:
            rospy.signal_shutdown('recorded 1M + 1K samples. stop now.')

    def haveTransform(self):
        if self.transform == None:
            try:
                self.transform = self.tfBuffer.lookup_transform('camera_depth_optical_frame', 'base_footprint',
                                                                rospy.Time())
                return True
            except (LookupException, ConnectivityException, ExtrapolationException):
                return False
        else:
            return True

    # def saveState(self, depthIm, modelStates, camInfo, transform):
        # state = GazeboState(header=depthIm.header, image=depthIm, camera_info=camInfo, model_states=modelStates,
        #                     transform=transform)
        # self.statePub.publish(state)

    def newScene(self):
        self.pauseService()
        # self.resetRobot()
        self.moveBarrels(self.num_barrels)
        self.unpauseService()

    def setPose(self, model_name, pose):
        ## Check if our model exists yet
        if (model_name in self.models.name):

            state = ModelState(model_name=model_name, pose=pose)

            response = self.modelStateService(state)

            if (response.success):
                rospy.loginfo("Successfully set model pose")
                return True

        rospy.loginfo("failed to set model pose")
        return False

    def resetRobot(self):
        self.setPose(self.robotName, self.robotPose)

    # Adapted from pips_test: gazebo_driver.py
    def spawn_barrel(self, model_name, initial_pose):
        # Must be unique in the gazebo world - failure otherwise
        # Spawning on top of something else leads to bizarre behavior
        model_path = os.path.expanduser("~/.gazebo/models/first_2015_trash_can/model.sdf")
        model_xml = load_model_xml(model_path)
        robot_namespace = rospy.get_namespace()
        gazebo_namespace = "/gazebo"
        reference_frame = ""

        success = gazebo_interface.spawn_sdf_model_client(model_name, model_xml,
                                                          robot_namespace, initial_pose, reference_frame,
                                                          gazebo_namespace)

    def moveBarrels(self, n):
        for i, xy in enumerate(self.barrel_points(self.minx, self.miny, self.maxx, self.maxy, self.grid_spacing, n)):
            print(i, xy)
            name = "barrel{}".format(i)
            print(name)

            pose = Pose()
            pose.position.x = xy[0]
            pose.position.y = xy[1]
            pose.orientation.w = 1

            if not self.setPose(name, pose):
                self.spawn_barrel(name, pose)

    def shutdown(self):
        self.unpauseService()
        self.resetWorldService()
        train_data = {'X': np.concatenate(self.egocircle[:-1000], 0), 'Y': np.concatenate(self.labels[:-1000], 0)}
        filename = 'lpips_train.pickle'
        pickle.dump(train_data, open(filename, 'wb'))
        print('training data saved')
        val_data = {'X': np.concatenate(self.egocircle[-1000:], 0), 'Y': np.concatenate(self.labels[-1000:], 0)}
        filename = 'lpips_val.pickle'
        pickle.dump(val_data, open(filename, 'wb'))
        print('val data saved')

    def run(self):
        # self.depthSub = rospy.Subscriber('image', Image, self.depthCallback, queue_size=self.queue_size)
        dr_controller = Client('egocircle_node', timeout=30)
        dr_controller.update_configuration({'max_depth': 6.0})
        self.egocircleSub = rospy.Subscriber('point_scan', LaserScan, self.ecCallback, queue_size=self.queue_size)
        rospy.spin()

    def reset(self):
        self.random.seed(self.seed)
        self.nprandom = np.random.RandomState(self.seed)

    def __init__(self, as_node=True, seed=None):
        if as_node:
            rospy.init_node('gazebo_state_recorder')
            # rospy.on_shutdown(self.shutdown)

            rospy.loginfo("gazebo_state_recorder node started")

        self.robotName = 'mobile_base'

        self.queue_size = 50

        self.num_barrels = 3
        self.minx = 1
        self.maxx = 3.0
        self.miny = -2.0
        self.maxy = 2.0
        self.grid_spacing = 1.0

        self.robotPose = Pose()
        self.robotPose.orientation.w = 1

        self.random = random.Random()
        self.seed = seed
        self.random.seed(seed)
        self.nprandom = np.random.RandomState(seed)

        self.models = None
        self.camInfo = None
        self.transform = None

        model_state_service_name = 'gazebo/set_model_state'
        pause_service_name = '/gazebo/pause_physics'
        unpause_service_name = '/gazebo/unpause_physics'

        self.tfBuffer = Buffer()
        self.tfListener = TransformListener(self.tfBuffer)

        rospy.loginfo("Waiting for service...")
        rospy.wait_for_service(model_state_service_name)
        self.modelStateService = rospy.ServiceProxy(model_state_service_name, SetModelState)
        rospy.loginfo("Service found...")

        rospy.wait_for_service(pause_service_name)
        self.pauseService = rospy.ServiceProxy(pause_service_name, Empty)
        rospy.loginfo("Service found...")

        self.resetWorldService = rospy.ServiceProxy('/gazebo/reset_world', std_srvs.srv.Empty)

        rospy.wait_for_service(unpause_service_name)
        self.unpauseService = rospy.ServiceProxy(unpause_service_name, Empty)
        rospy.loginfo("Service found...")

        self.stateSub = rospy.Subscriber('gazebo/model_states', ModelStates, self.statesCallback,
                                         queue_size=self.queue_size)

        self.camInfoSub = rospy.Subscriber('camera_info', CameraInfo, self.camInfoCallback, queue_size=self.queue_size)
        # self.statePub = rospy.Publisher('gazebo_data', GazeboState, queue_size=self.queue_size)
        self.egocircle = []
        self.labels = []
        self.resetWorldService()
        self.unpauseService()

        rospy.on_shutdown(self.shutdown)


if __name__ == '__main__':
    try:
        a = Prototype()
        a.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("exception")
