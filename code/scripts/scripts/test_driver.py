#!/usr/bin/env python
from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
import rospy
from copy import deepcopy

import actionlib
import torch
from move_base_msgs.msg import *
from geometry_msgs.msg import PoseWithCovarianceStamped
from pprint import pprint
import tf
from actionlib_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry, OccupancyGrid
from kobuki_msgs.msg import BumperEvent
import tf2_ros
import math
import std_srvs.srv as std_srvs
import dynamic_reconfigure.client
import numpy as np
from sensor_msgs.msg import LaserScan, Image
from rospy_message_converter.message_converter import convert_dictionary_to_ros_message as convert_dict
import math
from collections import deque
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Quaternion, Pose, Point, Vector3
from std_msgs.msg import Header, ColorRGBA
from cv_bridge import CvBridge
from skimage.transform import rescale
import rospy
import actionlib
from move_base_msgs.msg import *
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from pprint import pprint
import tf
from actionlib_msgs.msg import GoalStatus
from nav_msgs.msg import Odometry, Path, OccupancyGrid
from kobuki_msgs.msg import BumperEvent
import tf2_ros
import math
import std_srvs.srv as std_srvs
from sensor_msgs.msg import LaserScan, Image
import rosbag
import datetime
import os
from move_base_msgs.msg import MoveBaseActionGoal, MoveBaseActionFeedback
import threading
import time


# from dynamic_reconfigure.server import Server
# from /opt/ros/kinetic/share/teb_local_planner/cfg/TebLocalPlannerReconfigure.cfg import TebLocalPlannerReconfigure

class ResultRecorder(object):
    def __init__(self):
        self.lock = threading.Lock()
        self.vel_sub = rospy.Subscriber("navigation_velocity_smoother/raw_cmd_vel", Twist, self.twistCB, queue_size=1)
        self.scan_sub = rospy.Subscriber("/scan", LaserScan, self.scanCB, queue_size=1)
        self.egocircle_sub = rospy.Subscriber("/point_scan", LaserScan, self.egocircleCB, queue_size=1)

        self.global_plan_sub = rospy.Subscriber("/move_base/CustomizedNavfnROS/plan", Path, self.planCB, queue_size=1)
        # self.global_plan_sub = rospy.Subscriber("/move_base/NavfnROS/plan", Path, self.planCB, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.imageCB, queue_size=1)
        self.depth_sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.depthCB, queue_size=1)

        #self.goal_sub = rospy.Subscriber("move_base/goal", MoveBaseActionGoal, self.goalCB, queue_size=1)

        self.scan = None
        self.egocircle = None
        self.plan = None
        self.image = None
        self.depth = None
        self.feedback = None

        # bagpath = "~/simulation_data/maze_room/" + str(datetime.datetime.now()) + ".bag"
        # self.bagfilepath = os.path.expanduser(bagpath)
        # print "bag file = " + self.bagfilepath + "\n"
        # self.bagfile = rosbag.Bag(f=self.bagfilepath, mode='w', compression=rosbag.Compression.LZ4)
    '''
    def record(self, twist, scan, egocircle, plan, image, depth, feedback):
        self.lock.acquire()
        start_t = time.time()
        self.bagfile.write("scan", scan, plan.header.stamp)
        self.bagfile.write("egocircle", egocircle, plan.header.stamp)
        self.bagfile.write("global_plan", plan, plan.header.stamp)
        self.bagfile.write("image", image, plan.header.stamp)
        self.bagfile.write("depth", depth, plan.header.stamp)
        self.bagfile.write("cmd", twist, plan.header.stamp)
        self.bagfile.write("feedback", feedback, plan.header.stamp)
        self.lock.release()
        rospy.logdebug("Sample recorded! Took: " + str((time.time() - start_t)*1000) + "ms")
    '''


    def twistCB(self, data):
        rospy.logdebug("Command received!")

        # if(self.scan is not None and self.egocircle is not None and self.plan is not None and self.image is not None and self.depth is not None and self.feedback is not None):
        #     self.record(data, self.scan, self.egocircle, self.plan, self.image, self.depth, self.feedback)

    def scanCB(self, data):
        rospy.logdebug("Scan received!")

        self.lock.acquire()
        self.scan = data
        self.lock.release()
        rospy.logdebug("Scan updated!")

    def egocircleCB(self, data):
        rospy.logdebug("Egocircle received!")

        self.lock.acquire()
        self.egocircle = data
        self.lock.release()
        rospy.logdebug("Egocircle updated!")

    def planCB(self, data):
        rospy.logdebug("Plan received!")

        self.lock.acquire()
        self.plan = data
        self.lock.release()
        rospy.logdebug("Plan updated!")

    def imageCB(self, data):
        rospy.logdebug("Image received!")

        self.lock.acquire()
        self.image = data
        self.lock.release()
        rospy.logdebug("Image updated!")

    def depthCB(self, data):
        rospy.logdebug("Depth received!")

        self.lock.acquire()
        self.depth = data
        self.lock.release()
        rospy.logdebug("Depth updated!")


    def setGoal(self, data):
        rospy.logdebug("Goal received!")

        self.lock.acquire()
        # self.bagfile.write("goal", data, data.target_pose.header.stamp)
        self.lock.release()
        rospy.logdebug("Goal recorded!")

    def getGlobalMap(self):
        map = rospy.wait_for_message("map", OccupancyGrid)
        # groundtruth_map = rospy.wait_for_message("groundtruth/map", OccupancyGrid)

        self.lock.acquire()
        # self.bagfile.write("global_map", map, map.header.stamp)
        # self.bagfile.write("groundtruth_global_map", groundtruth_map, map.header.stamp)
        self.lock.release()
        rospy.loginfo("Global Map recorded!")


    def feedback_cb(self, data):
        rospy.logdebug("Pose received!")

        self.lock.acquire()
        self.feedback = data
        self.lock.release()
        rospy.logdebug("Pose recorded!")


    def done(self):
        rospy.logdebug("'Done' Commanded!")

        self.lock.acquire()
        self.vel_sub.unregister()
        self.scan_sub.unregister()
        self.egocircle_sub.unregister()
        self.global_plan_sub.unregister()
        self.image_sub.unregister()
        self.depth_sub.unregister()
        # self.bagfile.close()
        self.lock.release()
        rospy.logdebug("'Done' accomplished!")


class BumperChecker(object):
    def __init__(self):
        self.sub = rospy.Subscriber("mobile_base/events/bumper", BumperEvent, self.bumperCB, queue_size=5)
        self.collided = False

    def bumperCB(self, data):
        if data.state == BumperEvent.PRESSED:
            self.collided = True


class LaserScanSaver(object):
    def __init__(self, interval=15):
        self.sub = rospy.Subscriber("/scan", LaserScan, self.LaserScanCB)
        self.msgs = deque(maxlen=10)

    def LaserScanCB(self, data):
        # if self.counter % self.interval == 0:
        self.msgs.append(data)
        # rospy.loginfo(data)
        # self.counter += 1

    def retrieve(self):
        msg = self.msgs[-1]
        ranges = np.array(msg.ranges).reshape(1, -1)
        mask = np.isnan(ranges)
        ranges[mask] = msg.range_max + 1
        return ranges


class DepthSaver(object):
    def __init__(self, interval=15):
        self.sub = rospy.Subscriber("/camera/depth/image_raw", Image, self.DepthCB)
        self.msgs = deque(maxlen=10)
        self.bridge = CvBridge()

    def DepthCB(self, data):
        image = np.array(self.bridge.imgmsg_to_cv2(data, "passthrough"))
        image[np.isnan(image)] = 11.
        image_rescaled = np.array(rescale(image, 0.5, anti_aliasing=True, multichannel=False),
                                  dtype=np.float32)[np.newaxis, ...]
        self.msgs.append(image_rescaled)

    def retrieve(self):
        msg = self.msgs[-1]

        return msg


class PositionChecker(object):
    def __init__(self):
        # self.sub = rospy.Subscriber("/odom", Odometry, self.positionCB)
        # self.tf = tf.TransformListener()
        self.feedback_subscriber = rospy.Subscriber("/ground_truth/state", Odometry, self.feedbackCB,
                                                    queue_size=100)
        self.position = None
        self.pub = rospy.Publisher("/visualization_marker", Marker, queue_size=100)
        self.pub2 = rospy.Publisher("/visualization_marker_text", Marker, queue_size=100)

    # def check_position(self):
        # t = self.tf.getLatestCommonTime("/base_link", "/map")
        # t = rospy.Time.now()
        # self.position, _ = self.tf.lookupTransform("/base_link", "/map", t)
        # print(self.position)

    def feedbackCB(self, feedback):
        cur_pos = feedback.pose.pose.position
        self.position = [cur_pos.x, cur_pos.y]

    def publish_param(self):
        planner = rospy.get_param("/move_base/dynamic_reconfigure")
        lookahead = rospy.get_param(planner + '/max_global_plan_lookahead_dist')
        marker = Marker(
            type=Marker.CYLINDER,
            id=2050,
            lifetime=rospy.Duration(1.5),
            pose=Pose(Point(0., 0., 0.), Quaternion(0, 0, 0, 1)),
            scale=Vector3(lookahead * 2, lookahead * 2, 0.1),
            header=Header(frame_id='base_link'),
            color=ColorRGBA(0.0, 0.5, 0.5, 0.5))
        text = str(lookahead)
        text_marker = Marker(
            type=Marker.TEXT_VIEW_FACING,
            id=2051,
            lifetime=rospy.Duration(1.5),
            pose=Pose(Point(0., 0., 0.), Quaternion(0, 0, 0, 1)),
            scale=Vector3(6., 6., 6.),
            header=Header(frame_id='base_link'),
            color=ColorRGBA(.5, 0.0, .5, 0.5),
            text=text)
        self.pub.publish(marker)
        self.pub2.publish(text_marker)


# Not currently in use
class OdomChecker(object):
    def __init__(self):
        # self.odom_timer = rospy.Timer(period = rospy.Duration(1), callback = self.checkOdom)
        self.not_moving = False
        self.collided = False

    def checkOdom(self, event=None):
        try:
            print("timer callback")
            now = rospy.Time.now()
            past = now - rospy.Duration(5.0)
            trans = self.tfBuffer.lookup_transform_full(
                target_frame='odom',
                target_time=rospy.Time.now(),
                source_frame='base_footprint',
                source_time=past,
                fixed_frame='odom',
                timeout=rospy.Duration(1.0)
            )
            print(str(trans))
            displacement = math.sqrt(
                trans.transform.translation.x * trans.transform.translation.x + trans.transform.translation.y * trans.transform.translation.y)
            print("Odom displacement: " + str(displacement))
            if (displacement < .05):
                self.not_moving = True

            past = now - rospy.Duration(1.0)
            trans = self.tfBuffer.lookup_transform_full(
                target_frame='map',
                target_time=now,
                source_frame='odom',
                source_time=past,
                fixed_frame='odom',
                timeout=rospy.Duration(1.0)
            )
            print(str(trans))
            displacement = math.sqrt(
                trans.transform.translation.x * trans.transform.translation.x + trans.transform.translation.y * trans.transform.translation.y)
            print("map displacement: " + str(displacement))
            if (displacement > .1):
                self.collided = True


        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            print(e)
            pass


class OdomAccumulator(object):
    def __init__(self):
        self.feedback_subscriber = rospy.Subscriber("move_base/feedback", MoveBaseActionFeedback, self.feedbackCB,
                                                    queue_size=5)
        self.path_length = 0
        self.prev_msg = None

    def feedbackCB(self, feedback):
        if self.prev_msg is not None:
            prev_pos = self.prev_msg.feedback.base_position.pose.position
            cur_pos = feedback.feedback.base_position.pose.position

            deltaX = cur_pos.x - prev_pos.x
            deltaY = cur_pos.y - prev_pos.y

            displacement = math.sqrt(deltaX * deltaX + deltaY * deltaY)
            self.path_length += displacement

        self.prev_msg = feedback

    def getPathLength(self):
        return self.path_length


def run_testImpl(pose):
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    print("waiting for server")
    client.wait_for_server()
    print("Done!")

    # Create the goal point
    goal = MoveBaseGoal()
    goal.target_pose.pose.position.x = pose[0]
    goal.target_pose.pose.position.y = pose[1]
    goal.target_pose.pose.position.z = pose[2]
    quaternion = tf.transformations.quaternion_from_euler(pose[3], pose[4], pose[5])

    goal.target_pose.pose.orientation.x = quaternion[0]
    goal.target_pose.pose.orientation.y = quaternion[1]
    goal.target_pose.pose.orientation.z = quaternion[2]
    goal.target_pose.pose.orientation.w = quaternion[3]
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp = rospy.Time.now()

    # Send the goal!
    print("sending goal")
    client.send_goal(goal)
    print("waiting for result")
    client.wait_for_result(rospy.Duration(300))
    print("done!")

    # 3 means success, according to the documentation
    # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
    print("getting goal status")
    print(client.get_goal_status_text())
    print("done!")
    print("returning state number")
    return client.get_state() == 3


def reset_costmaps():
    service = rospy.ServiceProxy("move_base/clear_costmaps", std_srvs.Empty)
    service()

# '''
def run_test(goal_pose):
    # Get a node handle and start the move_base action server
    # init_pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=1)

    # init_pose = PoseWithCovarianceStamped()
    # init_pose.header.frame_id = 'map'
    # init_pose.header.stamp = rospy.Time.now()
    # init_pose.pose.pose.position.x = 0.0
    # init_pose.pose.pose.position.y = 0.0
    # init_pose.pose.pose.position.z = 0.0
    # init_pose.pose.pose.orientation.x = 0.0
    # init_pose.pose.pose.orientation.y = 0.0
    # init_pose.pose.pose.orientation.z = 0.0
    # init_pose.pose.pose.orientation.w = 1.0
    # init_pose.pose.covariance[0] = 0.1; # pos.x
    # init_pose.pose.covariance[7] = 0.1; # pos.y
    # init_pose.pose.covariance[14] = 1000000.0;
    # init_pose.pose.covariance[21] = 1000000.0;
    # init_pose.pose.covariance[28] = 1000000.0;
    # init_pose.pose.covariance[35] = 0.05; # orientation.z

    # init_pub.publish(init_pose)

    # Action client for sending position commands
    bumper_checker = BumperChecker()
    odom_checker = OdomChecker()
    odom_accumulator = OdomAccumulator()
    # laserscan_saver = LaserScanSaver()

    # client = actionlib.SimpleActionClient('global_planner', MoveBaseAction)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    print("waiting for server")
    client.wait_for_server()
    print("Done!")

    # print "waiting for server"
    # client.wait_for_server()
    # print "Done!"

    # Create the goal point

    goal = MoveBaseGoal()
    goal.target_pose = goal_pose
    goal.target_pose.header.stamp = rospy.Time.now()
    r = rospy.Rate(5)
    # rospy.sleep(1.)
    r.sleep()
    # Send the goal!
    print("sending goal")
    client.send_goal(goal)
    print("waiting for result")

    # r = rospy.Rate(5)

    start_time = rospy.Time.now()

    result = None

    keep_waiting = True
    counter = 0
    while keep_waiting:
        state = client.get_state()
        # print "State: " + str(state)
        if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
            keep_waiting = False
        elif state is GoalStatus.PREEMPTED:
            client.send_goal(goal)
        elif bumper_checker.collided:
            keep_waiting = False
            result = "BUMPER_COLLISION"
        elif odom_checker.collided:
            keep_waiting = False
            result = "OTHER_COLLISION"
        elif odom_checker.not_moving:
            keep_waiting = False
            result = "STUCK"
        elif (rospy.Time.now() - start_time > rospy.Duration(200)):
            keep_waiting = False
            result = "TIMED_OUT"
        else:
            r.sleep()
            # counter += 1
            # if counter % 100 == 0:
            # random = np.random.random()
            # candidate = ['teb_local_planner/TebLocalPlannerROS', 'dwa_local_planner/DWAPlannerROS']
            # dr.update_configuration({"base_local_planner": candidate[np.round(random).astype(np.int)]})
            # print(rospy.get_param(planner + "/base_local_planner"))

    task_time = rospy.Time.now() - start_time

    path_length = odom_accumulator.getPathLength()

    if result is None:
        # client.wait_for_result(rospy.Duration(45))
        print("done!")

        # 3 means success, according to the documentation
        # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
        print("getting goal status")
        print(client.get_goal_status_text())
        print("done!")
        print("returning state number")
        # return client.get_state() == 3
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            result = "SUCCEEDED"
        elif state == GoalStatus.ABORTED:
            result = "ABORTED"
        elif state == GoalStatus.LOST:
            result = "LOST"
        elif state == GoalStatus.REJECTED:
            result = "REJECTED"
        elif state == GoalStatus.ACTIVE:
            result = "TIMED_OUT"
        else:
            result = "UNKNOWN"

    return {'result': result, 'time': task_time, 'path_length': path_length}
'''
def run_test(goal_pose):
    # Get a node handle and start the move_base action server
    # init_pub = rospy.Publisher('initialpose', PoseWithCovarianceStamped, queue_size=1)

    # init_pose = PoseWithCovarianceStamped()
    # init_pose.header.frame_id = 'map'
    # init_pose.header.stamp = rospy.Time.now()
    # init_pose.pose.pose.position.x = 0.0
    # init_pose.pose.pose.position.y = 0.0
    # init_pose.pose.pose.position.z = 0.0
    # init_pose.pose.pose.orientation.x = 0.0
    # init_pose.pose.pose.orientation.y = 0.0
    # init_pose.pose.pose.orientation.z = 0.0
    # init_pose.pose.pose.orientation.w = 1.0
    # init_pose.pose.covariance[0] = 0.1; # pos.x
    # init_pose.pose.covariance[7] = 0.1; # pos.y
    # init_pose.pose.covariance[14] = 1000000.0;
    # init_pose.pose.covariance[21] = 1000000.0;
    # init_pose.pose.covariance[28] = 1000000.0;
    # init_pose.pose.covariance[35] = 0.05; # orientation.z

    #init_pub.publish(init_pose)

    # Action client for sending position commands
    bumper_checker = BumperChecker()
    odom_checker = OdomChecker()
    odom_accumulator = OdomAccumulator()

    result_recorder = ResultRecorder()

    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    print "waiting for server"
    client.wait_for_server()
    print "Done!"

    # Create the goal point
    goal = MoveBaseGoal()
    goal.target_pose = goal_pose
    goal.target_pose.header.stamp = rospy.Time.now()

    result_recorder.setGoal(goal)
    result_recorder.getGlobalMap()

    # Send the goal!
    print "sending goal"
    client.send_goal(goal, feedback_cb=result_recorder.feedback_cb)
    print "waiting for result"

    r = rospy.Rate(5)

    start_time = rospy.Time.now()

    result = None

    keep_waiting = True
    while keep_waiting:
        state = client.get_state()
        #print "State: " + str(state)
        if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
            keep_waiting = False
        elif bumper_checker.collided:
            keep_waiting = False
            result = "BUMPER_COLLISION"
        elif odom_checker.collided:
            keep_waiting = False
            result = "OTHER_COLLISION"
        elif odom_checker.not_moving:
            keep_waiting = False
            result = "STUCK"
        elif (rospy.Time.now() - start_time > rospy.Duration(600)):
            keep_waiting = False
            result = "TIMED_OUT"
        else:
            r.sleep()

    task_time = str(rospy.Time.now() - start_time)

    result_recorder.done()

    path_length = str(odom_accumulator.getPathLength())

    if result is None:
        #client.wait_for_result(rospy.Duration(45))
        print "done!"


        # 3 means success, according to the documentation
        # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
        print "getting goal status"
        print(client.get_goal_status_text())
        print "done!"
        print "returning state number"
        #return client.get_state() == 3
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            result = "SUCCEEDED"
        elif state == GoalStatus.ABORTED:
            result = "ABORTED"
        elif state == GoalStatus.LOST:
            result = "LOST"
        elif state == GoalStatus.REJECTED:
            result = "REJECTED"
        elif state == GoalStatus.ACTIVE:
            result = "TIMED_OUT"

        else:
            result = "UNKNOWN"

    return {'result': result, 'time': task_time, 'path_length': path_length}  #, 'bag_file_path': result_recorder.bagfilepath
# '''


def run_test_predict(goal_pose, models, ranges, predict_recorder, truth, density, suffix='classifier'):
    bumper_checker = BumperChecker()
    odom_checker = OdomChecker()
    odom_accumulator = OdomAccumulator()
    laserscan_saver = LaserScanSaver()
    position_checker = PositionChecker()
    planner = rospy.get_param("/move_base/dynamic_reconfigure")
    dr_controller = dynamic_reconfigure.client.Client(planner, timeout=30)
    dr_egocircle = dynamic_reconfigure.client.Client('/egocircle_node', timeout=30)
    # dr_costmap = dynamic_reconfigure.client.Client('/move_base/local_costmap', timeout=30)
    accuracy = {}
    for param in list(models.keys()):
        accuracy[param] = 0.
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    print("waiting for server")
    client.wait_for_server()
    print("Done!")

    # Create the goal point
    goal = MoveBaseGoal()
    goal.target_pose = goal_pose
    goal.target_pose.header.stamp = rospy.Time.now()

    r = rospy.Rate(5)
    r.sleep()

    # Send the goal!
    print("sending goal")
    client.send_goal(goal)
    print("waiting for result")

    start_time = rospy.Time.now()

    result = None

    keep_waiting = True
    counter = 0
    num = 0
    while keep_waiting:
        state = client.get_state()
        # print "State: " + str(state)
        if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
            keep_waiting = False
        elif bumper_checker.collided:
            keep_waiting = False
            result = "BUMPER_COLLISION"
        elif odom_checker.collided:
            keep_waiting = False
            result = "OTHER_COLLISION"
        elif odom_checker.not_moving:
            keep_waiting = False
            result = "STUCK"
        elif (rospy.Time.now() - start_time > rospy.Duration(600)):
            keep_waiting = False
            result = "TIMED_OUT"
        else:
            r.sleep()
            # position_checker.check_position()
            # position_checker.publish_param()
            counter += 1
            look_ahead = 5.
            if counter % 10 == 0:
                num += 1
                laserscan = laserscan_saver.retrieve()
                update = {}
                # update_costmap = False
                for param in list(models.keys()):
                    value = models[param].predict(laserscan)[0]
                    pos_x = position_checker.position[0]
                    pos_y = position_checker.position[1]
                    dx = int(np.clip((4 * (pos_x + 40)) // 80, 0, 4))
                    dy = int(np.clip((4 * (pos_y + 40)) // 80, 0, 4))
                    d = density[dx, dy]
                    # gt = truth[d][param]
                    gt = ranges[param][truth[d][param]]
                    if suffix == 'classifier':
                        value = ranges[param][value]
                    elif suffix == 'gt':
                        value = gt
                    accuracy[param] += (np.abs(gt - value) < 1.5)
                    if param == 'max_depth':
                        dr_egocircle.update_configuration({'max_depth': value})
                    else:
                        update[param] = value
                dr_controller.update_configuration(update)
                # print("updated params:")
                # print(update)
                predict_recorder.write(update)
                # random = np.random.random()
                # candidate = ['teb_local_planner/TebLocalPlannerROS', 'dwa_local_planner/DWAPlannerROS']
                # dr.update_configuration({"base_local_planner": candidate[np.round(random).astype(np.int)]})
                # print(rospy.get_param(planner + "/base_local_planner"))

    task_time = rospy.Time.now() - start_time

    path_length = odom_accumulator.getPathLength()
    for param in list(accuracy.keys()):
        accuracy[param] /= max(num, 1)
    if result is None:
        # client.wait_for_result(rospy.Duration(45))
        print("done!")

        # 3 means success, according to the documentation
        # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
        print("getting goal status")
        print(client.get_goal_status_text())
        print("done!")
        print("returning state number")
        # return client.get_state() == 3
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            result = "SUCCEEDED"
        elif state == GoalStatus.ABORTED:
            result = "ABORTED"
        elif state == GoalStatus.LOST:
            result = "LOST"
        elif state == GoalStatus.REJECTED:
            result = "REJECTED"
        elif state == GoalStatus.ACTIVE:
            result = "TIMED_OUT"

        else:
            result = "UNKNOWN"
    print("{} run, total predict times: {} with {}".format(result, num, accuracy))

    return {'result': result, 'time': task_time, 'path_length': path_length, 'accuracy': accuracy}


def run_test_dl(goal_pose, models, ranges, predict_recorder, truth, density, suffix='classifier'):

    # Action client for sending position commands
    bumper_checker = BumperChecker()
    odom_checker = OdomChecker()
    odom_accumulator = OdomAccumulator()
    laserscan_saver = LaserScanSaver()
    position_checker = PositionChecker()
    planner = rospy.get_param("/move_base/dynamic_reconfigure")
    dr_controller = dynamic_reconfigure.client.Client(planner, timeout=30)
    dr_egocircle = dynamic_reconfigure.client.Client('/egocircle_node', timeout=30)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    print("waiting for server")
    client.wait_for_server()
    print("Done!")

    # Create the goal point
    goal = MoveBaseGoal()
    goal.target_pose = goal_pose
    goal.target_pose.header.stamp = rospy.Time.now()

    r = rospy.Rate(5)
    r.sleep()

    # Send the goal!
    print("sending goal")
    client.send_goal(goal)
    print("waiting for result")

    start_time = rospy.Time.now()

    result = None

    keep_waiting = True
    counter = 0
    num = 0
    accuracy = {}
    for param in list(models.keys()):
        accuracy[param] = 0.
    while keep_waiting:
        state = client.get_state()
        # print "State: " + str(state)
        if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
            keep_waiting = False
        elif bumper_checker.collided:
            keep_waiting = False
            result = "BUMPER_COLLISION"
        elif odom_checker.collided:
            keep_waiting = False
            result = "OTHER_COLLISION"
        elif odom_checker.not_moving:
            keep_waiting = False
            result = "STUCK"
        elif (rospy.Time.now() - start_time > rospy.Duration(300)):
            keep_waiting = False
            result = "TIMED_OUT"
        else:
            r.sleep()
            counter += 1
            if counter > 10 and counter % 10 == 0:
                num += 1
                laserscan = laserscan_saver.retrieve()
                update = {}
                laserscan = torch.tensor(laserscan)
                laserscan = laserscan.unsqueeze(0)
                # laserscan = laserscan.unsqueeze(1)
                for param in list(models.keys()):
                    with torch.no_grad():
                        value = models[param](laserscan)
                    pos_x = position_checker.position[0]
                    pos_y = position_checker.position[1]
                    dx = int(np.clip((4 * (pos_x + 40)) // 80, 0, 4))
                    dy = int(np.clip((4 * (pos_y + 40)) // 80, 0, 4))
                    d = density[dx, dy]
                    gt = ranges[param][truth[d][param]]
                    if suffix == 'classifier':
                        value = torch.argmax(value, dim=-1).squeeze().item()
                        value = ranges[param][value]
                    elif suffix == 'regressor':
                        value = torch.squeeze(value).item()
                    elif suffix == 'gt':
                        value = gt
                    accuracy[param] += (np.abs(gt - value) < 1.5)
                    if param == 'max_depth':
                        # print 'egocircle ready'
                        # value = np.clip(value, 1.0, 5.5)
                        dr_egocircle.update_configuration({'max_depth': value})
                    else:
                        # value = np.clip(value, 0.0625, 2.0)
                        update[param] = value
                        # r.sleep()
                        # print update
                        dr_controller.update_configuration(update)
                predict_recorder.write(update)

    task_time = rospy.Time.now() - start_time

    path_length = odom_accumulator.getPathLength()
    for param in list(accuracy.keys()):
        accuracy[param] /= max(num, 1)
    if result is None:
        # client.wait_for_result(rospy.Duration(45))
        print("done!")

        # 3 means success, according to the documentation
        # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
        print("getting goal status")
        print(client.get_goal_status_text())
        print("done!")
        print("returning state number")
        # return client.get_state() == 3
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            result = "SUCCEEDED"
        elif state == GoalStatus.ABORTED:
            result = "ABORTED"
        elif state == GoalStatus.LOST:
            result = "LOST"
        elif state == GoalStatus.REJECTED:
            result = "REJECTED"
        elif state == GoalStatus.ACTIVE:
            result = "TIMED_OUT"

        else:
            result = "UNKNOWN"
    print("{} run, total predict times: {} with {}".format(result, num, accuracy))

    return {'result': result, 'time': task_time, 'path_length': path_length, 'accuracy': accuracy}


def run_test_dl_branch(goal_pose, models, ranges, predict_recorder, suffix='classifier'):

    # Action client for sending position commands
    bumper_checker = BumperChecker()
    odom_checker = OdomChecker()
    odom_accumulator = OdomAccumulator()
    laserscan_saver = LaserScanSaver()
    position_checker = PositionChecker()
    planner = rospy.get_param("/move_base/dynamic_reconfigure")
    dr_controller = dynamic_reconfigure.client.Client(planner, timeout=30)
    dr_egocircle = dynamic_reconfigure.client.Client('/egocircle_node', timeout=30)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    print("waiting for server")
    client.wait_for_server()
    print("Done!")

    # Create the goal point
    goal = MoveBaseGoal()
    goal.target_pose = goal_pose
    goal.target_pose.header.stamp = rospy.Time.now()

    r = rospy.Rate(5)
    r.sleep()

    # Send the goal!
    print("sending goal")
    client.send_goal(goal)
    print("waiting for result")

    start_time = rospy.Time.now()

    result = None

    keep_waiting = True
    counter = 0
    num = 0
    accuracy = {}
    p1 = 'max_depth'
    p2 = 'planner_frequency'
    for param in list(models.keys()):
        accuracy[param] = 0.
    while keep_waiting:
        state = client.get_state()
        # print "State: " + str(state)
        if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
            keep_waiting = False
        elif bumper_checker.collided:
            keep_waiting = False
            result = "BUMPER_COLLISION"
        elif odom_checker.collided:
            keep_waiting = False
            result = "OTHER_COLLISION"
        elif odom_checker.not_moving:
            keep_waiting = False
            result = "STUCK"
        elif (rospy.Time.now() - start_time > rospy.Duration(300)):
            keep_waiting = False
            result = "TIMED_OUT"
        else:
            r.sleep()
            counter += 1
            if counter > 10 and counter % 10 == 0:
                num += 1
                laserscan = laserscan_saver.retrieve()
                update = {}
                laserscan = torch.tensor(laserscan)
                laserscan = laserscan.unsqueeze(0)
                # laserscan = laserscan.unsqueeze(1)
                for param in list(models.keys()):
                    with torch.no_grad():
                        value1, value2 = models[param](laserscan)
                    if suffix == 'classifier':
                        value1 = torch.argmax(value1, dim=-1).squeeze().item()
                        value1 = ranges[p1][value1]
                        value2 = torch.argmax(value2, dim=-1).squeeze().item()
                        value2 = ranges[p2][value2]
                    elif suffix == 'regressor':
                        value1 = torch.squeeze(value1).item()
                        value2 = torch.squeeze(value2).item()
                    # if param == 'max_depth':
                        # print 'egocircle ready'
                        # value = np.clip(value, 1.0, 5.5)
                    dr_egocircle.update_configuration({'max_depth': value1})
                    # else:
                        # value = np.clip(value, 0.0625, 2.0)
                    update[p2] = value2
                        # r.sleep()
                        # print update
                    dr_controller.update_configuration(update)
                predict_recorder.write(update)

    task_time = rospy.Time.now() - start_time

    path_length = odom_accumulator.getPathLength()
    if result is None:
        # client.wait_for_result(rospy.Duration(45))
        print("done!")

        # 3 means success, according to the documentation
        # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
        print("getting goal status")
        print(client.get_goal_status_text())
        print("done!")
        print("returning state number")
        # return client.get_state() == 3
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            result = "SUCCEEDED"
        elif state == GoalStatus.ABORTED:
            result = "ABORTED"
        elif state == GoalStatus.LOST:
            result = "LOST"
        elif state == GoalStatus.REJECTED:
            result = "REJECTED"
        elif state == GoalStatus.ACTIVE:
            result = "TIMED_OUT"

        else:
            result = "UNKNOWN"
    print("{} run, total predict times: {}".format(result, num))

    return {'result': result, 'time': task_time, 'path_length': path_length, 'accuracy': accuracy}


def run_test_rl(goal_pose, models, ranges, param, predict_recorder, reward_recorder, shortest_path):
    shortest_path = np.array(shortest_path)
    bumper_checker = BumperChecker()
    odom_checker = OdomChecker()
    odom_accumulator = OdomAccumulator()
    laserscan_saver = LaserScanSaver()
    # laserscan_saver = DepthSaver()
    position_checker = PositionChecker()
    planner = rospy.get_param("/move_base/dynamic_reconfigure")
    dr_controller = dynamic_reconfigure.client.Client(planner, timeout=30)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    print("waiting for server")
    client.wait_for_server()
    print("Done!")

    # Create the goal point
    goal = MoveBaseGoal()
    goal.target_pose = goal_pose
    goal.target_pose.header.stamp = rospy.Time.now()

    r = rospy.Rate(5)
    r.sleep()

    # Send the goal!
    print("sending goal")
    client.send_goal(goal)
    print("waiting for result")
    r.sleep()

    start_time = rospy.Time.now()

    result = None

    keep_waiting = True
    counter = 0
    num = 0
    rewards = []
    laserscan0 = laserscan_saver.retrieve()
    # print(ranges[param])
    # print(np.argwhere(ranges[param] == 1))
    ind = 4  # np.argwhere(ranges[param] == 1)[0].item()
    while keep_waiting:
        state = client.get_state()
        # print "State: " + str(state)
        if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
            keep_waiting = False
        elif bumper_checker.collided:
            keep_waiting = False
            result = "BUMPER_COLLISION"
        elif odom_checker.collided:
            keep_waiting = False
            result = "OTHER_COLLISION"
        elif odom_checker.not_moving:
            keep_waiting = False
            result = "STUCK"
        elif (rospy.Time.now() - start_time > rospy.Duration(600)):
            keep_waiting = False
            result = "TIMED_OUT"
        else:
            counter += 1
            if counter % 10 == 0:
                num += 1
                update = {}
                # for param in models.keys():
                # position_checker.check_position()
                pos_x = position_checker.position[0]
                pos_y = position_checker.position[1]
                laserscan = laserscan_saver.retrieve()
                pos = np.array([pos_x, pos_y])
                # pos = pos[:, np.newaxis]
                dists = np.sum(np.square(shortest_path), axis=1) - 2 * np.matmul(shortest_path, pos) + np.sum(np.square(pos))
                reward = -np.sqrt(np.amin(dists))
                rewards.append(reward)
                experience = (laserscan0, ind, np.array([reward]), laserscan, 0.)
                models.remember(experience)
                with torch.no_grad():
                    ind = models.act(laserscan[np.newaxis, ...])
                value = ranges[param][ind]
                laserscan0 = deepcopy(laserscan)
                update[param] = value
                dr_controller.update_configuration(update)
                reward_recorder.write(reward)
                predict_recorder.write(update)
                if len(models.replay_buffer) > 32:
                    models.replay(32)
                    # print('update')
            else:
                r.sleep()

    task_time = rospy.Time.now() - start_time

    path_length = odom_accumulator.getPathLength()
    if result is None:
        # client.wait_for_result(rospy.Duration(45))
        print("done!")

        # 3 means success, according to the documentation
        # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
        print("getting goal status")
        print(client.get_goal_status_text())
        print("done!")
        print("returning state number")
        # return client.get_state() == 3
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            result = "SUCCEEDED"
        elif state == GoalStatus.ABORTED:
            result = "ABORTED"
        elif state == GoalStatus.LOST:
            result = "LOST"
        elif state == GoalStatus.REJECTED:
            result = "REJECTED"
        elif state == GoalStatus.ACTIVE:
            result = "TIMED_OUT"
        else:
            result = "UNKNOWN"

    if result == 'SUCCEEDED':
        reward = old_div(1000. * np.sum(np.sqrt(np.sum(np.square(shortest_path[1:, :]
                                                         - shortest_path[:-1, :]), axis=1))),path_length)
    else:
        reward = -1000.
    rewards.append(reward)
    reward_recorder.write(reward)
    laserscan = laserscan_saver.retrieve()
    # ind = models.act(laserscan)
    experience = (laserscan0, ind, np.array([reward]), laserscan, 1.)
    models.remember(experience)
    reward_sum = np.sum(rewards)
    print("{} run, total predict times: {} with {}".format(result, num, reward_sum))

    return {'result': result, 'time': task_time, 'path_length': path_length, 'rewards': reward_sum}


def run_test_rl_bc(goal_pose, models, ranges, param, predict_recorder, truth, density):
    bumper_checker = BumperChecker()
    odom_checker = OdomChecker()
    odom_accumulator = OdomAccumulator()
    laserscan_saver = LaserScanSaver()
    # laserscan_saver = DepthSaver()
    position_checker = PositionChecker()
    planner = rospy.get_param("/move_base/dynamic_reconfigure")
    dr_controller = dynamic_reconfigure.client.Client(planner, timeout=30)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    print("waiting for server")
    client.wait_for_server()
    print("Done!")

    # Create the goal point
    goal = MoveBaseGoal()
    goal.target_pose = goal_pose
    goal.target_pose.header.stamp = rospy.Time.now()

    r = rospy.Rate(5)
    r.sleep()

    # Send the goal!
    print("sending goal")
    client.send_goal(goal)
    print("waiting for result")
    r.sleep()

    start_time = rospy.Time.now()

    result = None

    keep_waiting = True
    counter = 0
    num = 0
    while keep_waiting:
        state = client.get_state()
        # print "State: " + str(state)
        if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
            keep_waiting = False
        elif bumper_checker.collided:
            keep_waiting = False
            result = "BUMPER_COLLISION"
        elif odom_checker.collided:
            keep_waiting = False
            result = "OTHER_COLLISION"
        elif odom_checker.not_moving:
            keep_waiting = False
            result = "STUCK"
        elif (rospy.Time.now() - start_time > rospy.Duration(600)):
            keep_waiting = False
            result = "TIMED_OUT"
        else:
            counter += 1
            if counter % 10 == 0:
                num += 1
                update = {}
                # for param in models.keys():
                # position_checker.check_position()
                pos_x = position_checker.position[0]
                pos_y = position_checker.position[1]
                laserscan = laserscan_saver.retrieve()
                # pos = pos[:, np.newaxis]
                dx = int((3 * (pos_x + 10)) // 20)
                dy = int((3 * (pos_y + 10)) // 20)
                d = density[dx, dy]
                gt = truth[d][param]
                experience = (laserscan, gt)
                models.remember(experience)
                value = ranges[param][gt]
                update[param] = value
                dr_controller.update_configuration(update)
                predict_recorder.write(update)
                if len(models.replay_buffer) > 32:
                    models.behavior_clone(32)
            else:
                r.sleep()

    task_time = rospy.Time.now() - start_time

    path_length = odom_accumulator.getPathLength()
    if result is None:
        # client.wait_for_result(rospy.Duration(45))
        print("done!")

        # 3 means success, according to the documentation
        # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
        print("getting goal status")
        print(client.get_goal_status_text())
        print("done!")
        print("returning state number")
        # return client.get_state() == 3
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            result = "SUCCEEDED"
        elif state == GoalStatus.ABORTED:
            result = "ABORTED"
        elif state == GoalStatus.LOST:
            result = "LOST"
        elif state == GoalStatus.REJECTED:
            result = "REJECTED"
        elif state == GoalStatus.ACTIVE:
            result = "TIMED_OUT"
        else:
            result = "UNKNOWN"

    print("{} run, total predict times: {}".format(result, num))

    return {'result': result, 'time': task_time, 'path_length': path_length}


def run_test_rl_aux(goal_pose, models, ranges, param, predict_recorder, reward_recorder, shortest_path, density):
    shortest_path = np.array(shortest_path)
    bumper_checker = BumperChecker()
    odom_checker = OdomChecker()
    odom_accumulator = OdomAccumulator()
    laserscan_saver = LaserScanSaver()
    # laserscan_saver = DepthSaver()
    position_checker = PositionChecker()
    planner = rospy.get_param("/move_base/dynamic_reconfigure")
    dr_controller = dynamic_reconfigure.client.Client(planner, timeout=30)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    print("waiting for server")
    client.wait_for_server()
    print("Done!")

    # Create the goal point
    goal = MoveBaseGoal()
    goal.target_pose = goal_pose
    goal.target_pose.header.stamp = rospy.Time.now()

    r = rospy.Rate(5)
    r.sleep()

    # Send the goal!
    print("sending goal")
    client.send_goal(goal)
    print("waiting for result")
    r.sleep()

    start_time = rospy.Time.now()

    result = None

    keep_waiting = True
    counter = 0
    num = 0
    rewards = []
    laserscan0 = laserscan_saver.retrieve()
    ind = np.argwhere(ranges[param] == 3.0)[0].item()
    while keep_waiting:
        state = client.get_state()
        # print "State: " + str(state)
        if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
            keep_waiting = False
        elif bumper_checker.collided:
            keep_waiting = False
            result = "BUMPER_COLLISION"
        elif odom_checker.collided:
            keep_waiting = False
            result = "OTHER_COLLISION"
        elif odom_checker.not_moving:
            keep_waiting = False
            result = "STUCK"
        elif (rospy.Time.now() - start_time > rospy.Duration(600)):
            keep_waiting = False
            result = "TIMED_OUT"
        else:
            counter += 1
            if counter % 10 == 0:
                num += 1
                update = {}
                # for param in models.keys():
                # position_checker.check_position()
                pos_x = position_checker.position[0]
                pos_y = position_checker.position[1]
                dx = int(np.clip((4 * (pos_x + 10)) // 20, 0, 4))
                dy = int(np.clip((4 * (pos_y + 10)) // 20, 0, 4))
                d = int((density[dx, dy] - 0.75)*4)
                laserscan = laserscan_saver.retrieve()
                pos = np.array([pos_x, pos_y])
                # pos = pos[:, np.newaxis]
                dists = np.sum(np.square(shortest_path), axis=1) - 2 * np.matmul(shortest_path, pos) + np.sum(np.square(pos))
                reward = -np.sqrt(np.amin(dists))
                rewards.append(reward)
                experience = (laserscan0, ind, np.array([reward]), laserscan, 0., d)
                models.remember(experience)
                with torch.no_grad():
                    ind = models.act(laserscan[np.newaxis, ...])
                value = ranges[param][ind]
                laserscan0 = deepcopy(laserscan)
                update[param] = value
                dr_controller.update_configuration(update)
                reward_recorder.write(reward)
                predict_recorder.write(update)
                if len(models.replay_buffer) > 32:
                    models.replay(32)
                    # print('update')
            else:
                r.sleep()

    task_time = rospy.Time.now() - start_time

    path_length = odom_accumulator.getPathLength()
    if result is None:
        # client.wait_for_result(rospy.Duration(45))
        print("done!")

        # 3 means success, according to the documentation
        # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
        print("getting goal status")
        print(client.get_goal_status_text())
        print("done!")
        print("returning state number")
        # return client.get_state() == 3
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            result = "SUCCEEDED"
        elif state == GoalStatus.ABORTED:
            result = "ABORTED"
        elif state == GoalStatus.LOST:
            result = "LOST"
        elif state == GoalStatus.REJECTED:
            result = "REJECTED"
        elif state == GoalStatus.ACTIVE:
            result = "TIMED_OUT"
        else:
            result = "UNKNOWN"

    if result == 'SUCCEEDED':
        reward = old_div(1000. * np.sum(np.sqrt(np.sum(np.square(shortest_path[1:, :]
                                                         - shortest_path[:-1, :]), axis=1))),path_length)
    else:
        reward = -1000.
    rewards.append(reward)
    reward_recorder.write(reward)
    pos_x = position_checker.position[0]
    pos_y = position_checker.position[1]
    dx = int(np.clip((4 * (pos_x + 10)) // 20, 0, 4))
    dy = int(np.clip((4 * (pos_y + 10)) // 20, 0, 4))
    d = int((density[dx, dy] - 0.75) * 4)
    laserscan = laserscan_saver.retrieve()
    # ind = models.act(laserscan)
    experience = (laserscan0, ind, np.array([reward]), laserscan, 1., d)
    models.remember(experience)
    reward_sum = np.sum(rewards)
    print("{} run, total predict times: {} with {}".format(result, num, reward_sum))

    return {'result': result, 'time': task_time, 'path_length': path_length, 'rewards': reward_sum}


def run_test_rl_bc_aux(goal_pose, models, ranges, param, predict_recorder, truth, density):
    bumper_checker = BumperChecker()
    odom_checker = OdomChecker()
    odom_accumulator = OdomAccumulator()
    laserscan_saver = DepthSaver()
    position_checker = PositionChecker()
    planner = rospy.get_param("/move_base/dynamic_reconfigure")
    dr_controller = dynamic_reconfigure.client.Client(planner, timeout=30)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    print("waiting for server")
    client.wait_for_server()
    print("Done!")

    # Create the goal point
    goal = MoveBaseGoal()
    goal.target_pose = goal_pose
    goal.target_pose.header.stamp = rospy.Time.now()

    r = rospy.Rate(5)
    r.sleep()

    # Send the goal!
    print("sending goal")
    client.send_goal(goal)
    print("waiting for result")
    r.sleep()

    start_time = rospy.Time.now()

    result = None

    keep_waiting = True
    counter = 0
    num = 0
    while keep_waiting:
        state = client.get_state()
        # print "State: " + str(state)
        if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
            keep_waiting = False
        elif bumper_checker.collided:
            keep_waiting = False
            result = "BUMPER_COLLISION"
        elif odom_checker.collided:
            keep_waiting = False
            result = "OTHER_COLLISION"
        elif odom_checker.not_moving:
            keep_waiting = False
            result = "STUCK"
        elif (rospy.Time.now() - start_time > rospy.Duration(600)):
            keep_waiting = False
            result = "TIMED_OUT"
        else:
            counter += 1
            if counter % 10 == 0:
                num += 1
                update = {}
                # for param in models.keys():
                # position_checker.check_position()
                pos_x = position_checker.position[0]
                pos_y = position_checker.position[1]
                laserscan = laserscan_saver.retrieve()
                # pos = pos[:, np.newaxis]
                dx = int((4 * (pos_x + 10)) // 20)
                dy = int((4 * (pos_y + 10)) // 20)
                d = int((density[dx, dy] - 0.75)*4)
                # d = density[dx, dy]
                gt = truth[d][param]
                experience = (laserscan, gt, d)
                models.remember(experience)
                value = ranges[param][gt]
                update[param] = value
                dr_controller.update_configuration(update)
                predict_recorder.write(update)
                if len(models.replay_buffer) > 32:
                    models.behavior_clone(32)
            else:
                r.sleep()

    task_time = rospy.Time.now() - start_time

    path_length = odom_accumulator.getPathLength()
    if result is None:
        # client.wait_for_result(rospy.Duration(45))
        print("done!")

        # 3 means success, according to the documentation
        # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
        print("getting goal status")
        print(client.get_goal_status_text())
        print("done!")
        print("returning state number")
        # return client.get_state() == 3
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            result = "SUCCEEDED"
        elif state == GoalStatus.ABORTED:
            result = "ABORTED"
        elif state == GoalStatus.LOST:
            result = "LOST"
        elif state == GoalStatus.REJECTED:
            result = "REJECTED"
        elif state == GoalStatus.ACTIVE:
            result = "TIMED_OUT"
        else:
            result = "UNKNOWN"

    print("{} run, total predict times: {}".format(result, num))

    return {'result': result, 'time': task_time, 'path_length': path_length}


def run_test_rl_double(goal_pose, models, ranges, param, predict_recorder, reward_recorder, shortest_path, density):
    shortest_path = np.array(shortest_path)
    bumper_checker = BumperChecker()
    odom_checker = OdomChecker()
    odom_accumulator = OdomAccumulator()
    laserscan_saver = LaserScanSaver()
    # laserscan_saver = DepthSaver()
    position_checker = PositionChecker()
    planner = rospy.get_param("/move_base/dynamic_reconfigure")
    dr_controller = dynamic_reconfigure.client.Client(planner, timeout=30)
    dr_egocircle = dynamic_reconfigure.client.Client('/egocircle_node', timeout=30)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    print("waiting for server")
    client.wait_for_server()
    print("Done!")

    # Create the goal point
    goal = MoveBaseGoal()
    goal.target_pose = goal_pose
    goal.target_pose.header.stamp = rospy.Time.now()

    r = rospy.Rate(5)
    r.sleep()

    # Send the goal!
    print("sending goal")
    client.send_goal(goal)
    print("waiting for result")
    r.sleep()

    start_time = rospy.Time.now()

    result = None

    keep_waiting = True
    counter = 0
    num = 0
    rewards = []
    laserscan0 = laserscan_saver.retrieve()
    ind = 4  # np.argwhere(ranges[param] == 3.0)[0].item()
    ind2 = 4
    while keep_waiting:
        state = client.get_state()
        # print "State: " + str(state)
        if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
            keep_waiting = False
        elif bumper_checker.collided:
            keep_waiting = False
            result = "BUMPER_COLLISION"
        elif odom_checker.collided:
            keep_waiting = False
            result = "OTHER_COLLISION"
        elif odom_checker.not_moving:
            keep_waiting = False
            result = "STUCK"
        elif (rospy.Time.now() - start_time > rospy.Duration(600)):
            keep_waiting = False
            result = "TIMED_OUT"
        else:
            counter += 1
            if counter % 10 == 0:
                num += 1
                update = {}
                # for param in models.keys():
                # position_checker.check_position()
                pos_x = position_checker.position[0]
                pos_y = position_checker.position[1]
                # dx = int(np.clip((4 * (pos_x + 10)) // 20, 0, 4))
                # dy = int(np.clip((4 * (pos_y + 10)) // 20, 0, 4))
                # d = int((density[dx, dy] - 0.75)*4)
                laserscan = laserscan_saver.retrieve()
                pos = np.array([pos_x, pos_y])
                # pos = pos[:, np.newaxis]
                dists = np.sum(np.square(shortest_path), axis=1) - 2 * np.matmul(shortest_path, pos) + np.sum(np.square(pos))
                reward = -np.sqrt(np.amin(dists))
                rewards.append(reward)
                experience = (laserscan0, ind, np.array([reward]), laserscan, 0., ind2)
                models.remember(experience)
                with torch.no_grad():
                    ind, ind2 = models.act(laserscan[np.newaxis, ...])
                value = ranges[param[0]][ind]
                value2 = ranges[param[1]][ind2]
                laserscan0 = deepcopy(laserscan)
                update[param[1]] = value2
                dr_controller.update_configuration(update)
                dr_egocircle.update_configuration({param[0]: value})
                update[param[0]] = value
                reward_recorder.write(reward)
                predict_recorder.write(update)
                if len(models.replay_buffer) > 32:
                    models.replay(32)
                    # print('update')
            else:
                r.sleep()

    task_time = rospy.Time.now() - start_time

    path_length = odom_accumulator.getPathLength()
    if result is None:
        # client.wait_for_result(rospy.Duration(45))
        print("done!")

        # 3 means success, according to the documentation
        # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
        print("getting goal status")
        print(client.get_goal_status_text())
        print("done!")
        print("returning state number")
        # return client.get_state() == 3
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            result = "SUCCEEDED"
        elif state == GoalStatus.ABORTED:
            result = "ABORTED"
        elif state == GoalStatus.LOST:
            result = "LOST"
        elif state == GoalStatus.REJECTED:
            result = "REJECTED"
        elif state == GoalStatus.ACTIVE:
            result = "TIMED_OUT"
        else:
            result = "UNKNOWN"

    if result == 'SUCCEEDED':
        reward = old_div(10000. * np.sum(np.sqrt(np.sum(np.square(shortest_path[1:, :]
                                                         - shortest_path[:-1, :]), axis=1))),path_length)
    else:
        reward = -10000.
    rewards.append(reward)
    reward_recorder.write(reward)
    pos_x = position_checker.position[0]
    pos_y = position_checker.position[1]
    # dx = int(np.clip((4 * (pos_x + 10)) // 20, 0, 4))
    # dy = int(np.clip((4 * (pos_y + 10)) // 20, 0, 4))
    # d = int((density[dx, dy] - 0.75) * 4)
    laserscan = laserscan_saver.retrieve()
    # ind = models.act(laserscan)
    experience = (laserscan0, ind, np.array([reward]), laserscan, 1., ind2)
    models.remember(experience)
    reward_sum = np.sum(rewards)
    print("{} run, total predict times: {} with {}".format(result, num, reward_sum))

    return {'result': result, 'time': task_time, 'path_length': path_length, 'rewards': reward_sum}


def run_test_rl_bc_double(goal_pose, models, ranges, param, predict_recorder, truth, density):
    bumper_checker = BumperChecker()
    odom_checker = OdomChecker()
    odom_accumulator = OdomAccumulator()
    # laserscan_saver = DepthSaver()
    laserscan_saver = LaserScanSaver()
    position_checker = PositionChecker()
    planner = rospy.get_param("/move_base/dynamic_reconfigure")
    dr_controller = dynamic_reconfigure.client.Client(planner, timeout=30)
    dr_egocircle = dynamic_reconfigure.client.Client('/egocircle_node', timeout=30)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    print("waiting for server")
    client.wait_for_server()
    print("Done!")

    # Create the goal point
    goal = MoveBaseGoal()
    goal.target_pose = goal_pose
    goal.target_pose.header.stamp = rospy.Time.now()

    r = rospy.Rate(5)
    r.sleep()

    # Send the goal!
    print("sending goal")
    client.send_goal(goal)
    print("waiting for result")
    r.sleep()

    start_time = rospy.Time.now()

    result = None

    keep_waiting = True
    counter = 0
    num = 0
    while keep_waiting:
        state = client.get_state()
        # print "State: " + str(state)
        if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
            keep_waiting = False
        elif bumper_checker.collided:
            keep_waiting = False
            result = "BUMPER_COLLISION"
        elif odom_checker.collided:
            keep_waiting = False
            result = "OTHER_COLLISION"
        elif odom_checker.not_moving:
            keep_waiting = False
            result = "STUCK"
        elif (rospy.Time.now() - start_time > rospy.Duration(600)):
            keep_waiting = False
            result = "TIMED_OUT"
        else:
            counter += 1
            if counter % 10 == 0:
                num += 1
                update = {}
                # for param in models.keys():
                # position_checker.check_position()
                pos_x = position_checker.position[0]
                pos_y = position_checker.position[1]
                laserscan = laserscan_saver.retrieve()
                # pos = pos[:, np.newaxis]
                dx = int((4 * (pos_x + 10)) // 20)
                dy = int((4 * (pos_y + 10)) // 20)
                # d = int((density[dx, dy] - 0.75)*4)
                d = density[dx, dy]
                gt = truth[d][param[0]]
                gt2 = truth[d][param[1]]
                experience = (laserscan, gt, gt2)
                models.remember(experience)
                value = ranges[param[1]][gt2]
                update[param[1]] = value
                dr_controller.update_configuration(update)
                value2 = ranges[param[0]][gt]
                update[param[0]] = value2
                dr_egocircle.update_configuration({param[0]: value2})
                predict_recorder.write(update)
                if len(models.replay_buffer) > 32:
                    models.behavior_clone(32)
            else:
                r.sleep()

    task_time = rospy.Time.now() - start_time

    path_length = odom_accumulator.getPathLength()
    if result is None:
        # client.wait_for_result(rospy.Duration(45))
        print("done!")

        # 3 means success, according to the documentation
        # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
        print("getting goal status")
        print(client.get_goal_status_text())
        print("done!")
        print("returning state number")
        # return client.get_state() == 3
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            result = "SUCCEEDED"
        elif state == GoalStatus.ABORTED:
            result = "ABORTED"
        elif state == GoalStatus.LOST:
            result = "LOST"
        elif state == GoalStatus.REJECTED:
            result = "REJECTED"
        elif state == GoalStatus.ACTIVE:
            result = "TIMED_OUT"
        else:
            result = "UNKNOWN"

    print("{} run, total predict times: {}".format(result, num))

    return {'result': result, 'time': task_time, 'path_length': path_length}


def run_test_rl_predict_double(goal_pose, models, ranges, param, predict_recorder):
    bumper_checker = BumperChecker()
    odom_checker = OdomChecker()
    odom_accumulator = OdomAccumulator()
    # laserscan_saver = DepthSaver()
    laserscan_saver = LaserScanSaver()
    # position_checker = PositionChecker()
    planner = rospy.get_param("/move_base/dynamic_reconfigure")
    dr_controller = dynamic_reconfigure.client.Client(planner, timeout=30)
    dr_egocircle = dynamic_reconfigure.client.Client('/egocircle_node', timeout=30)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    print("waiting for server")
    client.wait_for_server()
    print("Done!")

    # Create the goal point
    goal = MoveBaseGoal()
    goal.target_pose = goal_pose
    goal.target_pose.header.stamp = rospy.Time.now()

    r = rospy.Rate(5)
    r.sleep()

    # Send the goal!
    print("sending goal")
    client.send_goal(goal)
    print("waiting for result")

    start_time = rospy.Time.now()

    result = None

    keep_waiting = True
    counter = 0
    num = 0
    # predictions = {}
    # for p in param:
    #     predictions[p] = []
    while keep_waiting:
        state = client.get_state()
        if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
            keep_waiting = False
        elif bumper_checker.collided:
            keep_waiting = False
            result = "BUMPER_COLLISION"
        elif odom_checker.collided:
            keep_waiting = False
            result = "OTHER_COLLISION"
        elif odom_checker.not_moving:
            keep_waiting = False
            result = "STUCK"
        elif (rospy.Time.now() - start_time > rospy.Duration(600)):
            keep_waiting = False
            result = "TIMED_OUT"
        else:
            r.sleep()
            counter += 1
            if counter % 10 == 0:
                num += 1
                update = {}
                laserscan = laserscan_saver.retrieve()
                with torch.no_grad():
                    ind, ind2 = models.act(laserscan[np.newaxis, ...])
                value = ranges[param[1]][ind2]
                update[param[1]] = value
                # predictions[param[1]].append(value)
                dr_controller.update_configuration(update)
                value2 = ranges[param[0]][ind]
                update[param[0]] = value2
                # predictions[param[0]].append(value2)
                dr_egocircle.update_configuration({param[0]: value2})
                predict_recorder.write(update)

    task_time = rospy.Time.now() - start_time

    path_length = odom_accumulator.getPathLength()
    if result is None:
        # client.wait_for_result(rospy.Duration(45))
        print("done!")

        # 3 means success, according to the documentation
        # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
        print("getting goal status")
        print(client.get_goal_status_text())
        print("done!")
        print("returning state number")
        # return client.get_state() == 3
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            result = "SUCCEEDED"
        elif state == GoalStatus.ABORTED:
            result = "ABORTED"
        elif state == GoalStatus.LOST:
            result = "LOST"
        elif state == GoalStatus.REJECTED:
            result = "REJECTED"
        elif state == GoalStatus.ACTIVE:
            result = "TIMED_OUT"
        else:
            result = "UNKNOWN"

    print("{} run, total predict times: {}".format(result, num))

    return {'result': result, 'time': task_time, 'path_length': path_length}


def run_test_rl_multi(goal_pose, models, ranges, param, predict_recorder, reward_recorder, shortest_path):
    shortest_path = np.array(shortest_path)
    bumper_checker = BumperChecker()
    odom_checker = OdomChecker()
    odom_accumulator = OdomAccumulator()
    laserscan_saver = LaserScanSaver()
    # laserscan_saver = DepthSaver()
    position_checker = PositionChecker()
    # planner = rospy.get_param("/move_base/dynamic_reconfigure")
    dr_controller = dynamic_reconfigure.client.Client('/move_base', timeout=30)
    dr_teb = dynamic_reconfigure.client.Client('/move_base/TebLocalPlannerROS', timeout=30)
    dr_egocircle = dynamic_reconfigure.client.Client('/egocircle_node', timeout=30)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    print("waiting for server")
    client.wait_for_server()
    print("Done!")

    # Create the goal point
    goal = MoveBaseGoal()
    goal.target_pose = goal_pose
    goal.target_pose.header.stamp = rospy.Time.now()

    r = rospy.Rate(5)
    r.sleep()

    # Send the goal!
    print("sending goal")
    client.send_goal(goal)
    print("waiting for result")
    r.sleep()

    start_time = rospy.Time.now()

    result = None

    keep_waiting = True
    counter = 0
    num = 0
    rewards = []
    laserscan0 = laserscan_saver.retrieve()
    depth = 4  # np.argwhere(ranges[param] == 3.0)[0].item()
    freq = 4
    cost, block, prefer, inflation, pose = 2, 2, 2, 2, 2
    while keep_waiting:
        state = client.get_state()
        # print "State: " + str(state)
        if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
            keep_waiting = False
        elif bumper_checker.collided:
            keep_waiting = False
            result = "BUMPER_COLLISION"
        elif odom_checker.collided:
            keep_waiting = False
            result = "OTHER_COLLISION"
        elif odom_checker.not_moving:
            keep_waiting = False
            result = "STUCK"
        elif (rospy.Time.now() - start_time > rospy.Duration(600)):
            keep_waiting = False
            result = "TIMED_OUT"
        else:
            counter += 1
            if counter % 10 == 0:
                num += 1
                update = {}
                # for param in models.keys():
                # position_checker.check_position()
                pos_x = position_checker.position[0]
                pos_y = position_checker.position[1]
                # dx = int(np.clip((4 * (pos_x + 10)) // 20, 0, 4))
                # dy = int(np.clip((4 * (pos_y + 10)) // 20, 0, 4))
                # d = int((density[dx, dy] - 0.75)*4)
                laserscan = laserscan_saver.retrieve()
                pos = np.array([pos_x, pos_y])
                # pos = pos[:, np.newaxis]
                dists = np.sum(np.square(shortest_path), axis=1) - 2 * np.matmul(shortest_path, pos) + np.sum(np.square(pos))
                reward = -np.sqrt(np.amin(dists))
                rewards.append(reward)
                experience = (laserscan0, np.array([reward]), laserscan, 0., depth, freq, cost, block, prefer, inflation, pose)
                models.remember(experience)
                with torch.no_grad():
                    inds = models.act(laserscan[np.newaxis, ...])
                for i in range(2, len(inds)):
                    update[param[i]] = ranges[param[i]][inds[i]]
                depth, freq, cost, block, prefer, inflation, pose = inds
                depth_value = ranges[param[0]][depth]
                freq_value = ranges[param[1]][freq]
                dr_teb.update_configuration(update)
                dr_controller.update_configuration({param[1]: freq_value})
                dr_egocircle.update_configuration({param[0]: depth_value})
                update[param[0]] = depth_value
                update[param[1]] = freq_value
                reward_recorder.write(reward)
                predict_recorder.write(update)
                if len(models.replay_buffer) > 128:
                    models.replay(128)
                    # print('update')
                if models.iter%100 == 0:
                    models.update()
            else:
                r.sleep()

    task_time = rospy.Time.now() - start_time

    path_length = odom_accumulator.getPathLength()
    if result is None:
        # client.wait_for_result(rospy.Duration(45))
        print("done!")

        # 3 means success, according to the documentation
        # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
        print("getting goal status")
        print(client.get_goal_status_text())
        print("done!")
        print("returning state number")
        # return client.get_state() == 3
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            result = "SUCCEEDED"
        elif state == GoalStatus.ABORTED:
            result = "ABORTED"
        elif state == GoalStatus.LOST:
            result = "LOST"
        elif state == GoalStatus.REJECTED:
            result = "REJECTED"
        elif state == GoalStatus.ACTIVE:
            result = "TIMED_OUT"
        else:
            result = "UNKNOWN"

    if result == 'SUCCEEDED':
        reward = old_div(10000. * np.sum(np.sqrt(np.sum(np.square(shortest_path[1:, :]
                                                         - shortest_path[:-1, :]), axis=1))),path_length)
    else:
        reward = -10000.
    rewards.append(reward)
    reward_recorder.write(reward)
    # dx = int(np.clip((4 * (pos_x + 10)) // 20, 0, 4))
    # dy = int(np.clip((4 * (pos_y + 10)) // 20, 0, 4))
    # d = int((density[dx, dy] - 0.75) * 4)
    laserscan = laserscan_saver.retrieve()
    # ind = models.act(laserscan)
    experience = (laserscan0, np.array([reward]), laserscan, 1., depth, freq, cost, block, prefer, inflation, pose)
    models.remember(experience)
    reward_sum = np.sum(rewards)
    print("{} run, total predict times: {} with {}".format(result, num, reward_sum))

    return {'result': result, 'time': task_time, 'path_length': path_length, 'rewards': reward_sum}


def run_test_rl_predict_multi(goal_pose, models, ranges, param, predict_recorder):
    bumper_checker = BumperChecker()
    odom_checker = OdomChecker()
    odom_accumulator = OdomAccumulator()
    # laserscan_saver = DepthSaver()
    laserscan_saver = LaserScanSaver()
    # position_checker = PositionChecker()
    # planner = rospy.get_param("/move_base/dynamic_reconfigure")
    dr_controller = dynamic_reconfigure.client.Client('/move_base', timeout=30)
    dr_teb = dynamic_reconfigure.client.Client('/move_base/TebLocalPlannerROS', timeout=30)
    dr_egocircle = dynamic_reconfigure.client.Client('/egocircle_node', timeout=30)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    print("waiting for server")
    client.wait_for_server()
    print("Done!")

    # Create the goal point
    goal = MoveBaseGoal()
    goal.target_pose = goal_pose
    goal.target_pose.header.stamp = rospy.Time.now()

    r = rospy.Rate(5)
    # for _ in range(10):
    r.sleep()

    # Send the goal!
    print("sending goal")
    client.send_goal(goal)
    print("waiting for result")

    start_time = rospy.Time.now()

    result = None

    keep_waiting = True
    counter = 0
    num = 0
    # predictions = {}
    # for p in param:
    #     predictions[p] = []
    while keep_waiting:
        state = client.get_state()
        # print(rospy.Time.now() - start_time, rospy.Duration(300), rospy.Time.now() - start_time>rospy.Duration(1))
        if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
            keep_waiting = False
        elif bumper_checker.collided:
            keep_waiting = False
            result = "BUMPER_COLLISION"
        elif odom_checker.collided:
            keep_waiting = False
            result = "OTHER_COLLISION"
        elif odom_checker.not_moving:
            keep_waiting = False
            result = "STUCK"
        elif (rospy.Time.now() - start_time > rospy.Duration(300)):
            keep_waiting = False
            result = "TIMED_OUT"
        else:
            r.sleep()
            counter += 1
            if counter > 10 and counter % 10 == 0:
                num += 1
                update = {}
                laserscan = laserscan_saver.retrieve()
                with torch.no_grad():
                    inds = models.act(laserscan[np.newaxis, ...])
                for i in range(2, len(inds)):
                    update[param[i]] = ranges[param[i]][inds[i]]
                depth, freq, cost, block, prefer, inflation, pose = inds
                depth_value = ranges[param[0]][depth]
                freq_value = ranges[param[1]][freq]
                dr_teb.update_configuration(update)
                dr_controller.update_configuration({param[1]: freq_value})
                dr_egocircle.update_configuration({param[0]: depth_value})
                update[param[0]] = depth_value
                update[param[1]] = freq_value
                predict_recorder.write(update)

    task_time = rospy.Time.now() - start_time

    path_length = odom_accumulator.getPathLength()
    if result is None:
        # client.wait_for_result(rospy.Duration(45))
        print("done!")

        # 3 means success, according to the documentation
        # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
        print("getting goal status")
        print(client.get_goal_status_text())
        print("done!")
        print("returning state number")
        # return client.get_state() == 3
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            result = "SUCCEEDED"
        elif state == GoalStatus.ABORTED:
            result = "ABORTED"
        elif state == GoalStatus.LOST:
            result = "LOST"
        elif state == GoalStatus.REJECTED:
            result = "REJECTED"
        elif state == GoalStatus.ACTIVE:
            result = "TIMED_OUT"
        else:
            result = "UNKNOWN"

    print("{} run, total predict times: {}".format(result, num))

    return {'result': result, 'time': task_time, 'path_length': path_length}


def run_test_rl_predict(goal_pose, models, ranges, param, predict_recorder):
    bumper_checker = BumperChecker()
    odom_checker = OdomChecker()
    odom_accumulator = OdomAccumulator()
    # laserscan_saver = DepthSaver()
    laserscan_saver = LaserScanSaver()
    # position_checker = PositionChecker()
    planner = rospy.get_param("/move_base/dynamic_reconfigure")
    dr_controller = dynamic_reconfigure.client.Client(planner, timeout=30)
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)

    print("waiting for server")
    client.wait_for_server()
    print("Done!")

    # Create the goal point
    goal = MoveBaseGoal()
    goal.target_pose = goal_pose
    goal.target_pose.header.stamp = rospy.Time.now()

    r = rospy.Rate(5)
    r.sleep()

    # Send the goal!
    print("sending goal")
    client.send_goal(goal)
    print("waiting for result")

    start_time = rospy.Time.now()

    result = None

    keep_waiting = True
    counter = 0
    num = 0
    # predictions = dict()
    # predictions[param] = []
    while keep_waiting:
        state = client.get_state()
        if state is not GoalStatus.ACTIVE and state is not GoalStatus.PENDING:
            keep_waiting = False
        elif bumper_checker.collided:
            keep_waiting = False
            result = "BUMPER_COLLISION"
        elif odom_checker.collided:
            keep_waiting = False
            result = "OTHER_COLLISION"
        elif odom_checker.not_moving:
            keep_waiting = False
            result = "STUCK"
        elif (rospy.Time.now() - start_time > rospy.Duration(600)):
            keep_waiting = False
            result = "TIMED_OUT"
        else:
            r.sleep()
            counter += 1
            if counter % 10 == 0:
                num += 1
                update = {}
                laserscan = laserscan_saver.retrieve()
                with torch.no_grad():
                    ind = models.act(laserscan[np.newaxis, ...])
                # print(ranges[param], ind)
                value = ranges[param][ind]
                update[param] = value
                # predictions[param].append(value)
                dr_controller.update_configuration(update)
                predict_recorder.write(update)

    task_time = rospy.Time.now() - start_time

    path_length = odom_accumulator.getPathLength()
    if result is None:
        # client.wait_for_result(rospy.Duration(45))
        print("done!")

        # 3 means success, according to the documentation
        # http://docs.ros.org/api/actionlib_msgs/html/msg/GoalStatus.html
        print("getting goal status")
        print(client.get_goal_status_text())
        print("done!")
        print("returning state number")
        # return client.get_state() == 3
        state = client.get_state()
        if state == GoalStatus.SUCCEEDED:
            result = "SUCCEEDED"
        elif state == GoalStatus.ABORTED:
            result = "ABORTED"
        elif state == GoalStatus.LOST:
            result = "LOST"
        elif state == GoalStatus.REJECTED:
            result = "REJECTED"
        elif state == GoalStatus.ACTIVE:
            result = "TIMED_OUT"
        else:
            result = "UNKNOWN"

    print("{} run, total predict times: {}".format(result, num))

    return {'result': result, 'time': task_time, 'path_length': path_length}


if __name__ == "__main__":
    try:
        rospy.init_node('pips_test', anonymous=True)
        run_test()
    except rospy.ROSInterruptException:
        print("Keyboard Interrupt")
