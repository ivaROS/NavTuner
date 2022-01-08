#!/usr/bin/env python

from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import zip
from builtins import range
from builtins import object
from past.utils import old_div
import rospy
import random
import sys, os, time
from geometry_msgs.msg import PoseStamped, Pose, PoseArray, Point, Quaternion, Transform, TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from copy import deepcopy
import rospkg

from tf2_ros import TransformListener, Buffer, LookupException, ConnectivityException, ExtrapolationException, \
    StaticTransformBroadcaster
import tf
import math
# from pips_test import gazebo_driver

from gazebo_msgs.msg import ModelStates, ModelState, LinkState
from gazebo_msgs.srv import SetModelState, SetLinkState
from gazebo_msgs.srv import GetModelState
from gazebo_msgs.srv import DeleteModel

import numpy as np

from gazebo_ros import gazebo_interface
import std_srvs.srv as std_srvs

import std_msgs.msg as std_msgs


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


class GazeboDriver(object):
    # Copied from pips_test: gazebo_driver.py
    def rechoose_barrel_points(self, p, off, start, goal, D):
        barrels = p + off
        for b in barrels:
            dx = b[0] - start[0]
            dy = b[1] - start[1]
            d = np.sqrt(dx ** 2 + dy ** 2)
            if d < D:
                b[0] = old_div(dx * D, d) + start[0]
                b[1] = old_div(dy * D, d) + start[1]
            dx = b[0] - goal[0]
            dy = b[1] - goal[1]
            d = np.sqrt(dx ** 2 + dy ** 2)
            if d < D:
                b[0] += old_div(dx * D, d) + goal[0]
                b[1] += old_div(dy * D, d) + goal[1]
        return barrels - p

    def statesCallback(self, data):  # This comes in at ~100hz
        self.models = data

    def newScene(self):
        self.pause()
        self.resetRobot()
        self.moveBarrels(self.num_barrels)
        self.unpause()

    def setPose(self, model_name, pose):
        retval = False

        ## Check if our model exists yet
        # if (self.models is not None and model_name in self.models.name):

        try:
            state = ModelState(model_name=model_name, pose=pose)

            response = self.setModelState(state)

            # if (response.success):
            #     rospy.loginfo("Successfully set pose of " + str(model_name) + " to " + str(pose))
            #     retval = True
            # else:
            #     rospy.logwarn("Error setting model pose: " + str(response.status_message))
            #     retval = False
        except rospy.ServiceException as e:
            # rospy.logwarn("Error setting pose: " + str(e))
            # retval = False
            pass

        # time.sleep(.01)
        return retval

        # rospy.loginfo("failed to set model pose")
        # return False

    def setLink(self, model_name, pose):
        ## Check if our model exists yet
        if (self.models is not None and model_name in self.models.name):

            try:
                linkstate = LinkState(link_name=model_name + "::link", pose=pose)

                response = self.setLinkStateService(linkstate)

                # if (response.success):
                #     rospy.loginfo("Successfully set link state of " + str(model_name) + " to " + str(pose))
                #     return True
                # else:
                #     rospy.logwarn("Error setting link state: " + str(response.status_message))
            except rospy.ServiceException as e:
                pass
                # rospy.logwarn("Error setting pose: " + str(e))
            return False

    def pause(self):
        rospy.wait_for_service(self.pause_service_name, timeout=self.service_timeout)
        return self.pauseService()

    def unpause(self):
        rospy.wait_for_service(self.unpause_service_name, timeout=self.service_timeout)
        return self.unpauseService()

    def resetWorld(self):
        rospy.wait_for_service(self.reset_world_service_name, timeout=self.service_timeout)
        return self.resetWorldService()

    def setModelState(self, state):
        rospy.wait_for_service(self.set_model_state_service_name, timeout=self.service_timeout)
        return self.setModelStateService(state)

    def deleteModel(self, name):
        rospy.wait_for_service(self.delete_model_service_name, timeout=self.service_timeout)
        return self.deleteModelService(model_name=name)

    def resetRobotImpl(self, pose):
        self.pause()
        p = Pose()
        p.position.x = pose[0]
        p.position.y = pose[1]
        p.position.z = pose[2]
        quaternion = tf.transformations.quaternion_from_euler(pose[3], pose[4], pose[5])
        # print quaternion
        p.orientation.x = quaternion[0]
        p.orientation.y = quaternion[1]
        p.orientation.z = quaternion[2]
        p.orientation.w = quaternion[3]
        self.setPose('mobile_base', p)
        self.unpause()

    def resetOdom(self):
        self.odom_pub.publish()

    def moveRobot(self, pose):
        self.robotPose = pose
        self.setPose(self.robotName, pose)

    def resetBarrels(self, n):
        name = None
        for i in range(n):
            name = "barrel{}".format(i)
            pose = self.poses[i]
            self.setPose(name, pose)

    # Adapted from pips_test: gazebo_driver.py
    def spawn_barrel(self, model_name, initial_pose):
        # Must be unique in the gazebo world - failure otherwise
        # Spawning on top of something else leads to bizarre behavior
        # model_path = os.path.expanduser("~/.gazebo/models/first_2015_trash_can/model.sdf")
        # Fot book chapter
        # model_path = os.path.expanduser("~/.gazebo/models/drc_practice_blue_cylinder/model.sdf")
        path = self.rospack.get_path("nav_configs")
        # model_path = os.path.expanduser(path + "/models/box_lus.sdf")
        model_path = os.path.expanduser(path + "/models/box_lus.sdf")
        model_xml = load_model_xml(model_path)
        robot_namespace = rospy.get_namespace()
        gazebo_namespace = "/gazebo"
        reference_frame = ""

        success = gazebo_interface.spawn_sdf_model_client(model_name, model_xml,
                                                          robot_namespace, initial_pose, reference_frame,
                                                          gazebo_namespace)

    # Adapted from pips_test: gazebo_driver.py
    def spawn_obstacle(self, model_name, model_type, initial_pose):
        # Must be unique in the gazebo world - failure otherwise
        # Spawning on top of something else leads to bizarre behavior

        # model_filenames = {'box':'box_lus.sdf', 'cylinder':'cylinder.sdf'}
        model_filenames = {'box': 'box_lus.sdf', 'cylinder': 'cylinder.sdf', 'pole': 'pole_005_06.sdf',
                           'square_post': 'box_02_02_05.sdf'}

        if model_type not in model_filenames:
            rospy.logerr(
                "Model type [" + str(model_type) + "] is unknown! Known types are: ")  # TODO: print list of types
            return False

        model_xml = load_model_xml(self.model_path + model_filenames[model_type])
        robot_namespace = rospy.get_namespace()
        gazebo_namespace = "/gazebo"
        reference_frame = ""

        success = gazebo_interface.spawn_sdf_model_client(model_name, model_xml,
                                                          robot_namespace, initial_pose, reference_frame,
                                                          gazebo_namespace)

        # time.sleep(.1)
        return success

    def spawn_package_model(self, model_name, package_name, model_path, initial_pose):
        # Must be unique in the gazebo world - failure otherwise
        # Spawning on top of something else leads to bizarre behavior

        package_path = self.rospack.get_path(package_name)

        model_xml = load_model_xml(package_path + model_path)
        robot_namespace = rospy.get_namespace()
        gazebo_namespace = "/gazebo"
        reference_frame = ""

        success = gazebo_interface.spawn_sdf_model_client(model_name, model_xml,
                                                          robot_namespace, initial_pose, reference_frame,
                                                          gazebo_namespace)

        return success

    def spawn_local_database_model(self, model_name, model_type, initial_pose):
        # Must be unique in the gazebo world - failure otherwise
        # Spawning on top of something else leads to bizarre behavior

        package_path = os.path.expanduser("~/.gazebo/models/")
        model_path = model_type + "/model.sdf"

        model_xml = load_model_xml(package_path + model_path)
        robot_namespace = rospy.get_namespace()
        gazebo_namespace = "/gazebo"
        reference_frame = ""

        success = gazebo_interface.spawn_sdf_model_client(model_name, model_xml,
                                                          robot_namespace, initial_pose, reference_frame,
                                                          gazebo_namespace)

        return success

    def moveBarrelsTest(self, n, x, y):
        self.poses = []
        for i in range(n):
            name = "barrel{}".format(i)
            pose = Pose()
            pose.position.x = x[i]
            pose.position.y = y[i]
            pose.orientation.w = 1
            self.poses.append(pose)
            if not self.setPose(name, pose):
                self.spawn_barrel(name, pose)

    def moveBarrels(self, n, minx=None, miny=None, maxx=None, maxy=None, grid_spacing=None, random=False, region_num=3,
                    start=None, goal=None, d=None):
        self.poses = []

        minx = self.minx if minx is None else minx
        maxx = self.maxx if maxx is None else maxx
        miny = self.miny if miny is None else miny
        maxy = self.maxy if maxy is None else maxy
        grid_spacing = self.grid_spacing if grid_spacing is None else grid_spacing

        barrel_names = [name for name in self.models.name if "barrel" in name]

        for i, xy in enumerate(self.barrel_points(xmins=minx, ymins=miny, xmaxs=maxx, ymaxs=maxy, min_dist=grid_spacing,
                                                  num_barrels=n, random=random, region_num=region_num,
                                                  start=start, goal=goal, d=d)):
            # print i, xy
            name = "barrel{}".format(i)
            # print name

            if name in barrel_names: barrel_names.remove(name)

            pose = Pose()
            pose.position.x = xy[0]
            pose.position.y = xy[1]

            # random orientation
            angle = 2 * math.pi * self.random.uniform(0, 1)
            quaternion = tf.transformations.quaternion_from_euler(0, 0, angle)
            pose.orientation.x = quaternion[0]
            pose.orientation.y = quaternion[1]
            pose.orientation.z = quaternion[2]
            pose.orientation.w = quaternion[3]

            self.poses.append(pose)

            # print str(pose)

            if not self.setPose(name, pose):
                self.spawn_barrel(name, pose)

        for name in barrel_names:
            res = self.deleteModel(name=name)
            if not res.success:
                print(res.status_message)

    def moveObstacles(self, n, minx=None, miny=None, maxx=None, maxy=None, grid_spacing=None, random=False,
                      region_num=5,
                      model_types=['cylinder', 'box', 'pole', 'square_post'], start=None, goal=None, d=None, maze=None):  #
        self.poses = []

        minx = self.minx if minx is None else minx
        maxx = self.maxx if maxx is None else maxx
        miny = self.miny if miny is None else miny
        maxy = self.maxy if maxy is None else maxy
        grid_spacing = self.grid_spacing if grid_spacing is None else grid_spacing

        barrel_names = [name for name in self.models.name if "obstacle" in name]

        num_types = {}
        num_types['box'] = 0
        for model_type in model_types:
            num_types[model_type] = 0

        current_obstacles = [(self.models.name[i], self.models.pose[i]) for i in range(len(self.models.name)) if
                             "obstacle" in self.models.name[i]]

        for name, pose in current_obstacles:
            pose.position.z += 2

            self.setPose(name, pose)
            # self.setLink(name, pose)
        maze_barrels = None
        self.barrier = []
        if maze is not None:
            points = np.where(maze > 0)
            maze_barrels = list(np.array(list(zip(.49 * (points[0] + 1) - 10, .49 * (points[1] + 1) - 10))))
            maze_barrels = sorted(self.random.sample(maze_barrels, int(round(len(maze_barrels) * math.sqrt(math.sqrt(0.75 / d))))),
                             key=lambda x: (x[0], x[1]))
            for barrel in maze_barrels:
                if not (np.linalg.norm(barrel - start[0:2]) < 0.75
                    or np.linalg.norm(barrel - goal[0:2]) < 0.75):
                    model_type = 'box'
                    num_type = num_types[model_type]
                    num_types[model_type] += 1

                    name = model_type + "_obstacle{}".format(num_type)
                    pose = Pose()
                    pose.position.x = barrel[0]
                    pose.position.y = barrel[1]
                    pose.position.z = 0

                    pose.orientation.w = 1

                    self.poses.append(pose)

                    # print str(pose)

                    if name in barrel_names:
                        barrel_names.remove(name)
                        self.setPose(name, pose)
                    else:
                        self.barrier.append((barrel[0], barrel[1]))
                        self.spawn_obstacle(name, model_type, pose)

        for i, xy in enumerate(
                self.barrel_points(xmins=minx, ymins=miny, xmaxs=maxx, ymaxs=maxy, min_dist=grid_spacing,
                                   num_barrels=n, random=random, region_num=region_num,
                                   max_tries=10000, start=start, goal=goal, d=d, maze=maze_barrels)):
            # print i, xy

            model_type = self.random.choice(model_types)
            num_type = num_types[model_type]
            num_types[model_type] += 1

            name = model_type + "_obstacle{}".format(num_type)
            # print name

            pose = Pose()
            pose.position.x = xy[0]
            pose.position.y = xy[1]
            pose.position.z = 0

            pose.orientation.w = 1

            self.poses.append(pose)

            # print str(pose)

            if name in barrel_names:
                barrel_names.remove(name)
                self.setPose(name, pose)
            else:
                self.spawn_obstacle(name, model_type, pose)

        for name in barrel_names:
            print("Deleting: " + str(name))
            res = self.deleteModel(name=name)
            if not res.success:
                print(res.status_message)

    def barrel_points(self, xmins, ymins, xmaxs, ymaxs, min_dist, num_barrels, max_tries=2500, random=False,
                      region_num=5,
                      start=None, goal=None, d=None, random_scale=5., maze=None):
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

        xmaxs = np.array(xmaxs)
        xmins = np.array(xmins)
        ymaxs = np.array(ymaxs)
        ymins = np.array(ymins)

        barrels = []
        l = 0
        if maze is not None:
            # points = np.where(maze>0)
            # barrels = list(np.array(list(zip(0.5 * (points[0] + 1) - 10, 0.5 * (points[1] + 1) - 10))))
            barrels += maze
            l = len(barrels)

        if random:
            # '''
            xmin = np.amin(xmins)
            xmax = np.amax(xmaxs)
            ymin = np.amin(ymins)
            ymax = np.amax(ymaxs)
            # Get a dense grid of points
            grid_size_x = old_div((xmax - xmin), region_num)
            grid_size_y = old_div((ymax - ymin), region_num)
            grids = np.mgrid[xmin:xmax + .01:grid_size_x, ymin:ymax + .01:grid_size_y]
            # density = [144, 169, 196, 225, 256]  # 49, 64, 81, 100, 121, 144, 169, 196, 225, 256
            # '''
            density = [0.75, 1.0, 1.25, 1.5]
            # p = 1./np.array(density)
            # p /= np.sum(p)
            # p = np.ones(len(density)) / len(density)
            p = np.array([0.25, 0.25, 0.25, 0.25])
            density_list = np.random.choice(density, size=(region_num, region_num), p=p)
            self.density = density_list
            # barrels = []
            # '''
            for i in range(region_num):
                for j in range(region_num):
                    d = density_list[i, j]
                    n = 0
                    k = 0
                    xxmin = grids[0, i, j]
                    xxmax = grids[0, i + 1, j + 1]
                    yymin = grids[1, i, j]
                    yymax = grids[1, i + 1, j + 1]
                    depth = xxmax - xxmin
                    width = yymax - yymin
                    length = min(depth, width)
                    while n < round(old_div(num_barrels, (region_num ** 2))) and k < round(old_div(max_tries, (region_num ** 2))):
                        a = self.random.random()
                        b = self.random.random()
                        x = xxmin + a * length
                        y = yymin + b * length
                        point = np.array((x, y))

                        barrel_valid = True
                        for barrel in barrels:
                            if np.linalg.norm(point - barrel) < d:
                                barrel_valid = False
                                break
                        if np.linalg.norm(point - start[:2]) < d or np.linalg.norm(point - goal[:2]) < d:
                            barrel_valid = False

                        if barrel_valid:
                            n += 1
                            barrels.append(point)
                        k += 1
                    """
                    subregion_num = np.ceil(np.sqrt(np.amax(density)/(region_num ** 2)))
                    grid_size_xx = (xxmax - xxmin) / subregion_num
                    grid_size_yy = (yymax - yymin) / subregion_num
                    points = np.mgrid[xxmin:xxmax:grid_size_xx, yymin:yymax:grid_size_yy]
                    points = points.swapaxes(0, 2)
                    points = points.reshape(points.size / 2, 2)
                    n = np.shape(points)[0]
                    inds = np.random.choice(n, size=d, replace=False)
                    p = points[inds]
                    off_x = (2 * self.nprandom.rand(d, 1) - 1) * grid_size_xx / random_scale
                    off_y = (2 * self.nprandom.rand(d, 1) - 1) * grid_size_yy / random_scale
                    off = np.concatenate([off_x, off_y], axis=1)
                    if start is not None and goal is not None and dmin is not None:
                        off = self.rechoose_barrel_points(p, off, start, goal, dmin)
                    barrels.append(p+off)
                    """
            # '''
            # barrels = np.concatenate(barrels, axis=0)
        else:
            '''
            xmin = np.amin(xmins)
            xmax = np.amax(xmaxs)
            ymin = np.amin(ymins)
            ymax = np.amax(ymaxs)
            # Get a dense grid of points
            grid_size_x = (xmax - xmin) / np.sqrt(num_barrels)
            grid_size_y = (ymax - ymin) / np.sqrt(num_barrels)
            xs = np.arange(xmin, xmax, grid_size_x)
            ys = np.arange(ymin, ymax, grid_size_y)
            points = np.mgrid[xmin:xmax:grid_size_x, ymin:ymax:grid_size_y]
            points = points.swapaxes(0, 2)
            points = points.reshape(points.size / 2, 2)
            # Choose random indexes
            # idx = self.random.sample(range(points.shape[0]), num_barrels)
            # print idx

            # Generate offsets
            off_x = (2 * self.nprandom.rand(num_barrels, 1) - 1) * grid_size_x / random_scale
            off_y = (2 * self.nprandom.rand(num_barrels, 1) - 1) * grid_size_y / random_scale
            off = np.concatenate([off_x, off_y], axis=1)
            idx = np.arange(num_barrels)

            # Compute barrel points
            p = points[idx]
            if start is not None and goal is not None and dmin is not None:
                off = self.rechoose_barrel_points(p, off, start, goal, dmin)
            barrels = p + off
            # '''
            # '''
            region_weights = np.random.random(len(xmins))
            self.region_barrel = np.zeros_like(region_weights)
            region_weights = old_div(region_weights, (np.sum(region_weights)))
            region_inds = list(range(len(xmins)))
            sampled_regions = self.nprandom.choice(region_inds, replace=True, p=region_weights, size=max_tries)
            n = 0
            i = 0
            # barrels = []
            while n < num_barrels and i < max_tries:
                a = self.random.random()
                b = self.random.random()

                region_ind = sampled_regions[i]

                xmax = xmaxs[region_ind]
                xmin = xmins[region_ind]
                ymax = ymaxs[region_ind]
                ymin = ymins[region_ind]

                depth = xmax - xmin
                width = ymax - ymin

                length = min(depth, width)

                if a * length < depth and b * length < width:
                    x = xmin + a * length
                    y = ymin + b * length
                    point = np.array((x, y))

                    barrel_valid = True
                    for barrel in barrels:
                        if np.linalg.norm(point - barrel) < min_dist:
                            barrel_valid = False
                            break
                    if np.linalg.norm(point - start[:2]) < min_dist or np.linalg.norm(point - goal[:2]) < min_dist:
                        barrel_valid = False

                    if barrel_valid:
                        n += 1
                        barrels.append(point)
                        self.region_barrel[region_ind] += 1
                i += 1
                # '''

        for barrel in barrels[l:]:
            self.barrier.append((barrel[0], barrel[1]))
            yield barrel

    def shutdown(self):
        self.unpause()
        self.resetWorld()

    def run(self):
        rospy.spin()

    def reset(self, seed=None):
        if seed is not None:
            self.seed = seed
        self.random.seed(self.seed)
        self.nprandom = np.random.RandomState(self.seed)

    def getRandInt(self, lower, upper):
        a = list(range(lower, upper + 1))
        start = self.random.choice(a)
        a.remove(start)
        end = self.random.choice(a)
        output = [start, end]
        return output

    def updateModels(self, timeout=2):
        self.models = rospy.wait_for_message(self.model_state_topic_name, ModelStates, timeout=timeout)

    # TODO: make return value depend on results of checks
    def checkServicesTopics(self, timeout=2):
        self.updateModels(timeout)
        rospy.wait_for_service(self.get_model_state_service_name, timeout=timeout)
        rospy.wait_for_service(self.pause_service_name, timeout=timeout)
        rospy.wait_for_service(self.reset_world_service_name, timeout=timeout)
        rospy.wait_for_service(self.unpause_service_name, timeout=timeout)
        rospy.wait_for_service(self.delete_model_service_name, timeout=timeout)

    def __init__(self, as_node=True, seed=None):
        self.region_barrel = None
        self.density = None
        if as_node:
            rospy.init_node('gazebo_state_recorder')

            rospy.loginfo("gazebo_state_recorder node started")

        self.robotName = 'mobile_base'

        self.queue_size = 50
        self.num_barrels = 3
        self.minx = [-3.5]
        self.maxx = [0.5]
        self.miny = [1.0]
        self.maxy = [5.0]
        self.grid_spacing = 0.

        self.service_timeout = 2.0

        self.poses = []
        self.robotPose = Pose()

        self.random = random.Random()
        self.seed = 0 if seed is None else seed
        self.random.seed(self.seed)
        self.nprandom = np.random.RandomState(self.seed)

        self.odom_pub = rospy.Publisher(
            '/mobile_base/commands/reset_odometry', std_msgs.Empty, queue_size=1)

        self.rospack = rospkg.RosPack()

        self.models = None

        self.set_model_state_service_name = 'gazebo/set_model_state'
        self.set_link_state_service_name = 'gazebo/set_link_state'
        self.pause_service_name = 'gazebo/pause_physics'
        self.unpause_service_name = 'gazebo/unpause_physics'
        self.get_model_state_service_name = 'gazebo/get_model_state'
        self.reset_world_service_name = "gazebo/reset_world"
        self.delete_model_service_name = "gazebo/delete_model"

        self.model_state_topic_name = 'gazebo/model_states'

        # rospy.loginfo("Waiting for service...")
        # rospy.wait_for_service(self.get_model_state_service_name)
        self.setModelStateService = rospy.ServiceProxy(self.set_model_state_service_name, SetModelState)
        # rospy.loginfo("Service found...")

        self.setLinkStateService = rospy.ServiceProxy(self.set_link_state_service_name, SetLinkState)

        # rospy.wait_for_service(self.pause_service_name)
        self.pauseService = rospy.ServiceProxy(self.pause_service_name, std_srvs.Empty)
        # rospy.loginfo("Service found...")

        # rospy.wait_for_service(self.reset_world_service_name)
        self.resetWorldService = rospy.ServiceProxy(self.reset_world_service_name, std_srvs.Empty)
        # rospy.loginfo("Service found...")

        # rospy.wait_for_service(self.unpause_service_name)
        self.unpauseService = rospy.ServiceProxy(self.unpause_service_name, std_srvs.Empty)
        # rospy.loginfo("Service found...")

        # rospy.wait_for_service(self.delete_model_service_name)
        self.deleteModelService = rospy.ServiceProxy(self.delete_model_service_name, DeleteModel)
        # rospy.loginfo("Service found...")

        # self.stateSub = rospy.Subscriber(self.model_state_topic_name, ModelStates, self.statesCallback, queue_size=self.queue_size)

        # rospy.wait_for_message(self.model_state_topic_name, ModelStates)
        # self.statePub = rospy.Publisher('gazebo_data', GazeboState, queue_size=self.queue_size)

        # self.resetWorldService()
        # self.unpauseService()

        # rospy.on_shutdown(self.shutdown)

        path = self.rospack.get_path("nav_configs")
        self.model_path = path + "/models/"


if __name__ == '__main__':
    try:
        a = GazeboDriver()
        a.run()
    except rospy.ROSInterruptException:
        rospy.loginfo("exception")
