#!/usr/bin/env python
import rospkg

import tf
from geometry_msgs.msg import Pose, Quaternion, PoseStamped

import test_driver
from testing_scenarios import TestingScenarios
import rospy


seed = 0

num_barrels = 500
min_obstacle_spacing = 1

starting_pose = [-9,0,0]
goal = [9,4,0]

# book chapter testing
# task = {'seed':seed, 'scenario':'fourth_floor', 'num_obstacles':num_barrels, 'target_id':4, 'min_obstacle_spacing':min_obstacle_spacing}

# task = {'seed':seed, 'scenario':'campus', 'num_obstacles':num_barrels, 'min_obstacle_spacing':min_obstacle_spacing}

#task= {'scenario': 'fourth_floor', 'controller':'pips_egocylindrical_dwa', 'seed':seed, 'robot':'quadrotor', 'min_obstacle_spacing': min_obstacle_spacing, 'num_obstacles': num_barrels}
# task= {'scenario': 'fourth_floor', 'controller':'dwa', 'seed':seed, 'robot':'pioneer', 'min_obstacle_spacing': min_obstacle_spacing, 'num_obstacles': num_barrels}

# task= {'scenario': 'sector_laser', 'controller':'dwa', 'seed':seed, 'robot':'turtlebot', 'min_obstacle_spacing': min_obstacle_spacing, 'num_obstacles': num_barrels}
task= {'scenario': 'campus', 'controller':'teb', 'seed':seed, 'robot':'turtlebot', 'min_obstacle_spacing': min_obstacle_spacing, 'num_obstacles': num_barrels}
#
# task = {'seed':seed, 'scenario':'campus_obstacle', 'num_obstacles':num_barrels, 'min_obstacle_spacing':min_obstacle_spacing}
# task = {'seed':seed, 'scenario':'fourth_floor_obstacle', 'num_obstacles':num_barrels, 'min_obstacle_spacing':min_obstacle_spacing}

# task = {'seed':seed, 'scenario':'sector'}

# seed = 7
# task= {'scenario': 'dense', 'controller':'teb', 'seed':seed, 'robot':'turtlebot', 'min_obstacle_spacing':0.75}

#task = {'seed':seed, 'scenario':'full_fourth_floor_obstacle', 'num_obstacles':50, 'min_obstacle_spacing':min_obstacle_spacing}

# task = {'seed':seed, 'scenario':'full_sector_laser', 'min_obstacle_spacing':min_obstacle_spacing, 'num_obstacles':num_barrels}

# task = {'seed':seed, 'scenario':'full_campus_obstacle', 'min_obstacle_spacing':min_obstacle_spacing, 'num_obstacles':num_barrels}
# task = {'seed':seed, 'scenario':'full_fourth_floor_obstacle', 'num_obstacles':50, 'min_obstacle_spacing':min_obstacle_spacing}
# task= {'scenario': 'dense', 'controller':'teb', 'seed':seed, 'robot':'turtlebot', 'min_obstacle_spacing':0.5}

# task = {'seed':seed, 'scenario':'full_campus_obstacle', 'min_obstacle_spacing':min_obstacle_spacing, 'num_obstacles':num_barrels}

task= {'scenario': 'empty', 'controller':'ego_teb', 'seed':8, 'robot':'turtlebot', 'min_obstacle_spacing': 0.25, 'num_obstacles': 500}

seed = 9  # 1
# rospack = rospkg.RosPack()
# map_path = rospack.get_path("nav_configs")
# map_path += '/maps/mazes/maze_1.0_seed' + str(seed)
# freq = 0.125
task = {'scenario': 'maze', 'controller': 'ego_teb', 'seed': seed, 'robot': 'turtlebot',
        'maze_file': 'maze_1.25.pickle', 'min_obstacle_spacing': 0.75, 'num_obstacles': 200, 'use_maze': True}
rospy.init_node('test_driver', anonymous=True)

# planner = rospy.get_param("/move_base")
# dr_controller = dynamic_reconfigure.client.Client("/move_base", timeout=30)
# dr_controller.update_configuration({'planner_frequency': freq})
# filename = 'planner_freq_' + str(freq) + '.bag'
test_driver.reset_costmaps()

#rospy.Rate(1).sleep()

scenarios = TestingScenarios()
scenario = scenarios.getScenario(task)

# rospy.Rate(1).sleep()

import time
start_time = time.time()
scenario.setupScenario()
end_time = time.time()

print str(end_time-start_time)

# from result_recorder import PoseRecorder
# bag = rosbag.Bag(filename, 'w')
# recorder = PoseRecorder(bag, '/robot_pose', 'pose')
# recorder.start()
# pose = [-4.202, -3.420, 0.000]
# pose_msg = Pose()
# pose_msg.position.x = pose[0]
# pose_msg.position.y = pose[1]
#
# q = tf.transformations.quaternion_from_euler(0, 0, pose[2])
# # msg = Quaternion(*q)
#
# pose_msg.orientation = Quaternion(*q)
#
# scenario.gazebo_driver.moveRobot(pose_msg)
#
# pose = [-1.590, -0.464, 0.000]
# pose_msg = Pose()
# pose_msg.position.x = pose[0]
# pose_msg.position.y = pose[1]
#
# q = tf.transformations.quaternion_from_euler(0, 0, pose[2])
# # msg = Quaternion(*q)
#
# pose_msg.orientation = Quaternion(*q)
# pose_stamped = PoseStamped()
# pose_stamped.pose = pose_msg
# pose_stamped.header.frame_id = "map"
# result = test_driver.run_test(goal_pose=scenario.getGoal())
# recorder.close()
# bag.close()
# print(result)