from __future__ import print_function
from __future__ import division
from builtins import str
from builtins import zip
from builtins import range
from past.utils import old_div
from builtins import object
import rospkg
import time

from gazebo_driver_v2 import GazeboDriver
from geometry_msgs.msg import Pose, PoseStamped, Quaternion
import numpy as np
import random
import tf
import math
from make_maze import Maze
import pickle
from a_star import AStarGraph, AStarSearch


class TestingScenarios(object):
    def __init__(self):
        self.gazebo_driver = GazeboDriver(as_node=False)

    def getScenario(self, task):
        if "seed" in task:
            self.gazebo_driver = GazeboDriver(as_node=False, seed=task["seed"])
        if "scenario" in task:
            scenario_type = task["scenario"]
            if scenario_type == "trashcans":
                return TrashCanScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "empty":
                return EmptyScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "empty_predict":
                return EmptyPredictScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "maze":
                return MazeScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "maze_predict":
                return MazePredictScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "maze_rl":
                return MazeRLScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "fourth_floor":
                return FourthFloorScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "fourth_floor_obstacle":
                return FourthFloorObstacleScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "fourth_floor_predict":
                return FourthFloorPredictScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "full_fourth_floor_obstacle":
                return FullFourthFloorObstacleScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "stereo_fourth_floor_obstacle":
                return StereoFourthFloorObstacleScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "campus":
                return CampusScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "campus_obstacle":
                return CampusObstacleScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "campus_predict":
                return CampusPredictScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "full_campus_obstacle":
                return FullCampusObstacleScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "stereo_campus_obstacle":
                return StereoCampusObstacleScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "sector":
                return SectorScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "sector_predict":
                return SectorPredictScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "sector_laser":
                return SectorLaserScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "full_sector_laser":
                return FullSectorLaserScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "stereo_sector_laser":
                return StereoSectorLaserScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "sector_extra":
                return SectorExtraScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "full_sector_extra":
                return FullSectorExtraScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "stereo_sector_extra":
                return StereoSectorExtraScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "sparse":
                return SparseScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "dense":
                return DenseScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "medium":
                return MediumScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "corridor_zigzag":
                return CorridorZigzagScenario(task=task, gazebo_driver=self.gazebo_driver)
            elif scenario_type == "corridor_zigzag_door":
                return CorridorZigzagDoorScenario(task=task, gazebo_driver=self.gazebo_driver)
            else:
                print("Error! Unknown scenario type [" + scenario_type + "]")
                return None
        elif "init_pose" in task and "goal" in task and "world" in task:
            return TestingScenario(task["world"], task["init_pose"], task["goal"], self.gazebo_driver)
        else:
            return None

    @staticmethod
    def getScenarioTypes():
        scenarios = [TrashCanScenario, SectorScenario, CampusScenario, FourthFloorObstacleScenario,
                     CampusObstacleScenario, SectorExtraScenario]
        return scenarios

    @staticmethod
    def getFieldNames():
        fieldnames = ["scenario"]
        for scenario in TestingScenarios.getScenarioTypes():
            fieldnames.extend(scenario.getUniqueFieldNames())
        return fieldnames


class TestingScenario(object):
    def __init__(self, world, init_pose, target_pose, gazebo_driver):
        self.gazebo_driver = gazebo_driver
        self.world = world
        self.init_pose = init_pose
        self.target_pose = target_pose

    def getGazeboLaunchFile(self, robot, world_file=None):
        if world_file is None:
            rospack = rospkg.RosPack()
            path = rospack.get_path("nav_configs")
            file = path + "/launch/gazebo_" + robot + "_" + self.world + "_world.launch"
        else:
            file = world_file
        return file

    def getStartingPose(self):
        return self.getPoseMsg(self.init_pose)

    def getGoal(self):
        goal = PoseStamped()
        goal.pose = self.getPoseMsg(self.target_pose)
        goal.header.frame_id = "map"
        return goal

    def setupScenario(self):
        print("Resetting robot...")
        # TODO: Check if reset successful; if not, wait briefly and try again,
        # eventually fail and throw error
        self.gazebo_driver.checkServicesTopics(10)

        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.unpause()

    @staticmethod
    def getUniqueFieldNames():
        return [""]

    def getPoseMsg(self, pose):
        pose_msg = Pose()
        pose_msg.position.x = pose[0]
        pose_msg.position.y = pose[1]

        q = tf.transformations.quaternion_from_euler(0, 0, pose[2])
        # msg = Quaternion(*q)

        pose_msg.orientation = Quaternion(*q)

        return pose_msg


class TrashCanScenario(TestingScenario):
    def __init__(self, task, gazebo_driver):
        self.gazebo_driver = gazebo_driver

        self.world = "rectangular"

        self.seed = task["seed"] if "seed" in task else 0
        self.num_barrels = task["num_obstacles"] if "num_obstacles" in task else 0

        self.init_pose = Pose()
        self.init_pose.position.x = 2
        self.init_pose.position.y = 3
        self.init_pose.orientation.x = 0
        self.init_pose.orientation.y = 0
        self.init_pose.orientation.z = 1
        self.init_pose.orientation.w = 0

        self.target_pose = PoseStamped()
        self.target_pose.pose.position.x = -5.0
        self.target_pose.pose.position.y = 3.0
        self.target_pose.pose.orientation.x = 0.0
        self.target_pose.pose.orientation.y = 0.0
        self.target_pose.pose.orientation.z = 0
        self.target_pose.pose.orientation.w = 1
        self.target_pose.header.frame_id = 'map'

    @staticmethod
    def getUniqueFieldNames():
        return ["num_obstacles", "seed"]

    def setupScenario(self):
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.init_pose)
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        self.gazebo_driver.moveBarrels(self.num_barrels)
        self.gazebo_driver.unpause()


class CampusScenario(TestingScenario):
    def __init__(self, task, gazebo_driver):
        self.gazebo_driver = gazebo_driver

        self.world = "campus"

        self.seed = task["seed"] if "seed" in task else 0
        self.num_barrels = task["num_obstacles"] if "num_obstacles" in task else 0

        self.init_id = task["init_id"] if "init_id" in task else None

        self.min_spacing = task["min_obstacle_spacing"] if "min_obstacle_spacing" in task else 0

        self.target_id = task["target_id"] if "target_id" in task else None

        self.random = random.Random()
        self.random.seed(self.seed)

        self.init_pose = Pose()
        self.init_pose.position.x = -12
        self.init_pose.position.y = 4
        self.init_pose.orientation.x = 0
        self.init_pose.orientation.y = 0
        self.init_pose.orientation.z = 0
        self.init_pose.orientation.w = 1

        self.target_poses = [[-12, 4, 1.57], [13, -10, 1.57], [13, 9.4, 1.57],
                             [-5.15, -9.25, 1.57], [-13.5, 15.25, 0],
                             [5.5, 6.5, -1.57], [1.5, 2, -1.57], [-4.0, -9.0, 0.],
                             [0.5, -10., 0.], [13., 14., 0.]]
        self.init_poses = [[-13, .5, 0], [-10., 15., 1.57], [-3., -5., -1.57],
                           [-2., 0., 0.], [-14., 4., 1.57], [-3., 15., -1.57],
                           [12., -12., 0.], [5., -6., 1.57], [12., 0., -1.57],
                           [-4., 3., 0.]]
        # self.init_poses = [[-12,4,1.57], [13,-10,1.57], [13,9.4,1.57], [-5.15,-9.25,1.57], [-13.5,15.25,0],
        #                    [5.5,6.5,-1.57], [1.5,2,-1.57], [-13,.5,0]]

        if self.init_id is None:
            self.init_id = self.random.randint(0, len(self.init_poses) - 1)
            task["init_id"] = self.init_id
        if self.target_id is None:
            self.target_id = self.random.randint(0, len(self.target_poses) - 1)
            task["target_id"] = self.target_id

        self.target_pose = PoseStamped()
        self.target_pose.pose.position.x = 4
        self.target_pose.pose.position.y = 16
        self.target_pose.pose.orientation.x = 0.0
        self.target_pose.pose.orientation.y = 0.0
        self.target_pose.pose.orientation.z = 0.0
        self.target_pose.pose.orientation.w = 1.0
        self.target_pose.header.frame_id = 'map'

        Zone1 = [[-14.21, 9.98], [-5, 6]]
        Zone2 = [[-6.1, 15.5], [-0.2, 12.96]]
        Zone3 = [[-4.61, 7.75], [-0.97, 5.08]]
        Zone4 = [[-3.64, 3.53], [-0.96, 1.707]]
        Zone5 = [[-3.96, 0.68], [-1, -1]]
        Zone6 = [[-5, -2], [0, -5]]
        Zone7 = [[-9.77, -5.18], [-5.46, -7.01]]
        Zone8 = [[-10.95, -0.85], [-8.59, -1.54]]
        Zone9 = [[9.49, 4], [13.45, -6.83]]
        Zone10 = [[0.64, -5.94], [6, -11.27]]
        Zone11 = [[3.61, 2.76], [7, 0]]
        Zone12 = [[0.61, -0.25], [3.94, -2.07]]
        Zone13 = [[0.34, 11.7], [11.65, 10.65]]

        zones = [Zone1, Zone2, Zone3, Zone4, Zone5, Zone6, Zone7, Zone8, Zone9, Zone10, Zone11, Zone12, Zone13]

        zones = np.swapaxes(zones, 0, 2)
        self.minx = zones[0][0]
        self.maxx = zones[0][1]
        self.maxy = zones[1][0]
        self.miny = zones[1][1]
        pass

    @staticmethod
    def getUniqueFieldNames():
        return ["num_obstacles", "seed", "target_id", "init_id", "min_obstacle_spacing"]

    def getPoseMsg(self, pose):
        pose_msg = Pose()
        pose_msg.position.x = pose[0]
        pose_msg.position.y = pose[1]

        q = tf.transformations.quaternion_from_euler(0, 0, pose[2])
        # msg = Quaternion(*q)

        pose_msg.orientation = Quaternion(*q)

        return pose_msg

    def getStartingPose(self):
        pose = self.init_poses[self.init_id]
        init_pose = self.getPoseMsg(pose=pose)

        return init_pose

    def getGoal(self):
        pose = self.target_poses[self.target_id]
        # x, y, t = self.init_poses[self.init_id]
        # t = (2 * np.random.random() - 1) * 3.14
        # dx = 5 * np.cos(t)
        # dy = 5 * np.sin(t)
        # pose = [x+dx, y+dy, t]
        pose_msg = self.getPoseMsg(pose=pose)
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose_msg
        pose_stamped.header.frame_id = "map"
        return pose_stamped

    def setupScenario(self):
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        self.gazebo_driver.moveBarrels(self.num_barrels, minx=self.minx, maxx=self.maxx, miny=self.miny, maxy=self.maxy,
                                       grid_spacing=self.min_spacing,
                                       start=self.init_poses[self.init_id], goal=self.target_poses[self.target_id])
        self.gazebo_driver.unpause()


class CampusObstacleScenario(CampusScenario):
    def __init__(self, task, gazebo_driver):
        super(CampusObstacleScenario, self).__init__(task=task, gazebo_driver=gazebo_driver)

        self.world = "campus_obstacle"

        self.num_obstacles = task["num_obstacles"] if "num_obstacles" in task else 500

    # def getUniqueFieldNames():
    #     return ["num_obstacles", "seed", "target_id", "init_id", "min_obstacle_spacing"]

    def setupScenario(self):
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        self.gazebo_driver.moveObstacles(self.num_obstacles, minx=self.minx, maxx=self.maxx, miny=self.miny,
                                         maxy=self.maxy, grid_spacing=self.min_spacing,
                                         start=self.init_poses[self.init_id], goal=self.target_poses[self.target_id])
        self.gazebo_driver.unpause()


class CampusPredictScenario(CampusObstacleScenario):
    def setupScenario(self):
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        # x, y, t = self.init_poses[self.init_id]
        # dx = 5 * np.cos(t)
        # dy = 5 * np.sin(t)
        # pose = [x + dx, y + dy, t]
        self.gazebo_driver.moveObstacles(self.num_obstacles, minx=self.minx, maxx=self.maxx, miny=self.miny,
                                         maxy=self.maxy, grid_spacing=self.min_spacing,
                                         start=self.init_poses[self.init_id], goal=self.target_poses[self.target_id],
                                         random=True)  #, model_types=['cylinder', 'square_post', 'pole']
        self.density = self.gazebo_driver.density
        self.gazebo_driver.unpause()


class FullCampusObstacleScenario(CampusObstacleScenario):
    def __init__(self, task, gazebo_driver):
        super(FullCampusObstacleScenario, self).__init__(task=task, gazebo_driver=gazebo_driver)

        self.world = "full_campus_obstacle"


class StereoCampusObstacleScenario(CampusObstacleScenario):
    def __init__(self, task, gazebo_driver):
        super(StereoCampusObstacleScenario, self).__init__(task=task, gazebo_driver=gazebo_driver)

        self.world = "stereo_campus_obstacle"


class SectorScenario(TestingScenario):
    def __init__(self, task, gazebo_driver, region_num=5, random=False):
        self.gazebo_driver = gazebo_driver

        self.world = "sector"

        self.seed = task["seed"] if "seed" in task else 0

        self.poses = [[-9, 9, -.78], [-9, 0, 0], [-9, -9, .78], [9, -9, 2.36], [9, 0, 3.14], [9, 9, -2.36],
                      [0, -9, 1.57]]  #
        # self.init_id = task["init_id"] if "init_id" in task else 0
        #
        # self.target_id = task["target_id"] if "target_id" in task else (self.init_id + len(self.poses) / 2) % len(
        #     self.poses)
        self.random = np.random
        self.random.seed(self.seed)

        self.num_barrels = task["num_obstacles"] if "num_obstacles" in task else 0

        self.min_spacing = task["min_obstacle_spacing"] if "min_obstacle_spacing" in task else None

        # Zone1 = [[-8.5, 9.5], [8.5, -9.5]]
        # zones = [Zone1]
        '''
        xs = np.linspace(-8.5, 9.5, region_num + 1)
        ys = np.linspace(8.5, -9.5, region_num + 1)
        zones = [[[xs[i], xs[i + 1]], [ys[j], ys[j + 1]]] for i in range(region_num) for j in range(region_num)]
        zones = np.swapaxes(zones, 0, 2)
        '''
        xmax = 8.5
        xmin = - xmax
        ymax = 9.5
        ymin = -ymax

        # Zone1 = [[-8.5, 9.5], [8.5, -9.5]]
        # zones = [Zone1]
        xs = np.linspace(xmin, xmax, region_num + 1)
        ys = np.linspace(ymax, ymin, region_num + 1)
        zones = [[[xs[i], xs[i + 1]], [ys[j], ys[j + 1]]] for i in range(region_num) for j in range(region_num)]
        zones = np.swapaxes(zones, 0, 2)

        x1 = zones[0][0]
        x2 = zones[0][1]
        y1 = zones[1][0]
        y2 = zones[1][1]

        self.minx = np.minimum(x1, x2)
        self.maxx = np.maximum(x1, x2)
        self.maxy = np.maximum(y1, y2)
        self.miny = np.minimum(y1, y2)

    @staticmethod
    def getUniqueFieldNames():
        return ["num_obstacles", "seed", "target_id", "init_id"]

    def getPoseMsg(self, pose):
        pose_msg = Pose()
        pose_msg.position.x = pose[0]
        pose_msg.position.y = pose[1]

        q = tf.transformations.quaternion_from_euler(0, 0, pose[2])
        # msg = Quaternion(*q)

        pose_msg.orientation = Quaternion(*q)

        return pose_msg

    def getStartingPose(self):
        # x = (2 * self.random.random() - 1) * 4
        # y = (2 * self.random.random() - 1) * 4
        # t = (2 * self.random.random() - 1) * 3.14
        # pose = [x, y, t]
        self.init_id = self.random.randint(0, len(self.poses) - 1)
        self.start = self.poses[self.init_id]
        init_pose = self.getPoseMsg(pose=self.start)

        return init_pose

    def setGoal(self):
        # x = (2 * self.random.random() - 1) * 9
        # y = self.random.random() * 18 - 9
        # t = (2 * self.random.random() - 1) * 3.14
        # pose = [x, y, t]
        self.target_id = self.random.randint(0, len(self.poses) - 1)
        while self.target_id == self.init_id:
            self.target_id = self.random.randint(0, len(self.poses) - 1)
        # x, y, t = self.start
        # dx = 5 * np.cos(t)
        # dy = 5 * np.sin(t)
        # pose = [x + dx, y + dy, t]
        self.goal = self.poses[self.target_id]

    def getGoal(self):
        init_pose = self.getPoseMsg(pose=self.goal)
        pose_stamped = PoseStamped()
        pose_stamped.pose = init_pose
        pose_stamped.header.frame_id = "map"
        return pose_stamped

    def setupScenario(self):
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.setGoal()
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        self.gazebo_driver.moveObstacles(self.num_barrels, minx=self.minx, maxx=self.maxx, miny=self.miny, maxy=self.maxy,
                                       grid_spacing=self.min_spacing,
                                         start=self.start, goal=self.goal)
        self.gazebo_driver.unpause()


class SectorPredictScenario(SectorScenario):
    def setupScenario(self):
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.setGoal()
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        # x, y, t = self.start
        # dx = 5 * np.cos(t)
        # dy = 5 * np.sin(t)
        # pose = [x + dx, y + dy, t]
        self.gazebo_driver.moveObstacles(self.num_barrels, minx=self.minx, maxx=self.maxx, miny=self.miny, maxy=self.maxy,
                                       grid_spacing=self.min_spacing, random=True,
                                         start=self.start, goal=self.goal)  # , model_types=['cylinder', 'square_post', 'pole']
        self.density = self.gazebo_driver.density
        self.gazebo_driver.unpause()


class EmptyScenario(TestingScenario):
    def __init__(self, task, gazebo_driver, region_num=5):
        self.gazebo_driver = gazebo_driver

        self.world = "empty"

        self.seed = task["seed"] if "seed" in task else 0
        self.use_maze = task["use_maze"] if "use_maze" in task else False
        rospack = rospkg.RosPack()
        map_path = rospack.get_path("nav_configs")
        map_path += '/maps/mazes/maze_1.0_seed' + str(self.seed)
        self.maze = Maze(7, 7, self.seed, path=map_path, hm=20., wm=20.).walls if self.use_maze else None
        pickle.dump(self.maze, open('maze_1.25_2.pickle', 'wb'))

        self.poses = [[-9, 9, -.78], [-9, 0, 0], [-9, -9, .78], [9, -9, 2.36], [9, 0, 3.14], [9, 9, -2.36],
                      [0, -9, 1.57]]  #

        self.init_id = task["init_id"] if "init_id" in task else 0

        self.target_id = task["target_id"] if "target_id" in task else (self.init_id + old_div(len(self.poses), 2)) % len(
            self.poses)

        self.random = np.random
        self.random.seed(self.seed)

        self.num_barrels = task["num_obstacles"] if "num_obstacles" in task else 0

        xmax = 10
        xmin = - xmax
        ymax = xmax
        ymin = xmin

        self.min_spacing = task["min_obstacle_spacing"] if "min_obstacle_spacing" in task \
            else old_div(old_div(min(xmax - xmin, ymax - ymin), np.sqrt(self.num_barrels)), 2) - 0.55

        # Zone1 = [[-8.5, 9.5], [8.5, -9.5]]
        # zones = [Zone1]
        xs = np.linspace(xmin, xmax, region_num + 1)
        ys = np.linspace(ymax, ymin, region_num + 1)
        zones = [[[xs[i], xs[i + 1]], [ys[j], ys[j + 1]]] for i in range(region_num) for j in range(region_num)]
        zones = np.swapaxes(zones, 0, 2)
        self.region_num = region_num

        x1 = zones[0][0]
        x2 = zones[1][0]
        y1 = zones[0][1]
        y2 = zones[1][1]

        self.minx = np.minimum(x1, x2)
        self.maxx = np.maximum(x1, x2)
        self.maxy = np.maximum(y1, y2)
        self.miny = np.minimum(y1, y2)

    @staticmethod
    def getUniqueFieldNames():
        return ["num_obstacles", "seed", "target_id", "init_id"]

    def getPoseMsg(self, pose):
        pose_msg = Pose()
        pose_msg.position.x = pose[0]
        pose_msg.position.y = pose[1]

        q = tf.transformations.quaternion_from_euler(0, 0, pose[2])
        # msg = Quaternion(*q)

        pose_msg.orientation = Quaternion(*q)

        return pose_msg

    def getStartingPose(self):
        # x = (2 * self.random.random() - 1) * 9.5
        # y = self.random.random() * 19 - 9.5
        # t = (2 * self.random.random() - 1) * 3.14
        # pose = [x, y, t]
        # init_pose = self.getPoseMsg(pose=pose)

        return self.init_pose

    def setStartingPose(self):
        x = (2 * self.random.random() - 1) * 9.5
        y = self.random.random() * 19 - 9.5
        t = (2 * self.random.random() - 1) * 3.14
        pose = [x, y, t]
        self.t = t
        self.start_pose = pose
        init_pose = self.getPoseMsg(pose=pose)
        self.init_pose = init_pose

    def getGoal(self):
        # x = (2 * self.random.random() - 1) * 9
        # y = self.random.random() * 18 - 9
        # t = (2 * self.random.random() - 1) * 3.14
        # pose = [x, y, t]
        # init_pose = self.getPoseMsg(pose=pose)
        # pose_stamped = PoseStamped()
        # pose_stamped.pose = init_pose
        # pose_stamped.header.frame_id = "map"
        # return pose_stamped
        return self.goal

    def setGoal(self):
        while True:
            x = (2 * self.random.random() - 1) * 9
            y = self.random.random() * 18 - 9
            t = (2 * self.random.random() - 1) * 3.14
            pose = [x, y, t]
            d = np.sqrt((x - self.start_pose[0]) ** 2 + (y - self.start_pose[1]) ** 2)
            if d >= 5.:
                break
        t = (2 * self.random.random() - 1) * 3.14/2
        self.goal_pose = pose
        init_pose = self.getPoseMsg(pose=pose)
        pose_stamped = PoseStamped()
        pose_stamped.pose = init_pose
        pose_stamped.header.frame_id = "map"
        self.goal = pose_stamped

    def setupScenario(self):
        self.setStartingPose()
        self.setGoal()
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        self.gazebo_driver.moveObstacles(self.num_barrels, minx=self.minx, maxx=self.maxx, miny=self.miny,
                                         maxy=self.maxy,
                                         grid_spacing=self.min_spacing, start=self.start_pose, goal=self.goal_pose,
                                         d=self.min_spacing, maze=self.maze)
        self.gazebo_driver.unpause()


class EmptyPredictScenario(EmptyScenario):
    def __init__(self, task, gazebo_driver, region_num=5, random=True):
        super(EmptyPredictScenario, self).__init__(task, gazebo_driver, region_num)
        self.region_barrel = None
        self.random_barrel = random

    def setupScenario(self):
        self.setStartingPose()
        self.setGoal()
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        self.gazebo_driver.moveObstacles(self.num_barrels, minx=self.minx, maxx=self.maxx, miny=self.miny,
                                         maxy=self.maxy,
                                         grid_spacing=self.min_spacing, random=self.random_barrel,
                                         region_num=self.region_num,
                                         start=self.start_pose, goal=self.goal_pose, d=self.min_spacing)
        self.density = self.gazebo_driver.density
        self.region_barrel = self.gazebo_driver.region_barrel
        self.gazebo_driver.unpause()


class MazeScenario(TestingScenario):
    def __init__(self, task, gazebo_driver, region_num=5):
        self.gazebo_driver = gazebo_driver

        self.world = "maze"

        self.seed = task["seed"] if "seed" in task else 0
        rospack = rospkg.RosPack()
        maze_file = task['maze_file'] if 'maze_file' in task else 'maze_1.25.pickle'
        self.maze = pickle.load(open(maze_file, 'rb'), encoding='latin1')
        self.use_maze = True
        points = np.where(self.maze > 0)
        self.maze_barrels = list(np.array(list(zip(.49 * (points[0] + 1) - 10, .49 * (points[1] + 1) - 10))))

        self.poses = [[-9, 9, -.78], [-9, 0, 0], [-9, -9, .78], [9, -9, 2.36], [9, 0, 3.14], [9, 9, -2.36],
                      [0, -9, 1.57]]  #

        self.init_id = task["init_id"] if "init_id" in task else 0

        self.target_id = task["target_id"] if "target_id" in task else (self.init_id + old_div(len(self.poses), 2)) % len(
            self.poses)

        self.random = np.random
        self.random.seed(self.seed)

        self.num_barrels = task["num_obstacles"] if "num_obstacles" in task else 0

        xmax = 9.75
        xmin = - xmax
        ymax = xmax
        ymin = xmin

        self.min_spacing = task["min_obstacle_spacing"] if "min_obstacle_spacing" in task \
            else old_div(old_div(min(xmax - xmin, ymax - ymin), np.sqrt(self.num_barrels)), 2) - 0.55

        # Zone1 = [[-8.5, 9.5], [8.5, -9.5]]
        # zones = [Zone1]
        xs = np.linspace(xmin, xmax, region_num + 1)
        ys = np.linspace(ymax, ymin, region_num + 1)
        zones = [[[xs[i], xs[i + 1]], [ys[j], ys[j + 1]]] for i in range(region_num) for j in range(region_num)]
        zones = np.swapaxes(zones, 0, 2)
        self.region_num = region_num

        x1 = zones[0][0]
        x2 = zones[1][0]
        y1 = zones[0][1]
        y2 = zones[1][1]

        self.minx = np.minimum(x1, x2)
        self.maxx = np.maximum(x1, x2)
        self.maxy = np.maximum(y1, y2)
        self.miny = np.minimum(y1, y2)

    @staticmethod
    def getUniqueFieldNames():
        return ["num_obstacles", "seed", "target_id", "init_id"]

    def getPoseMsg(self, pose):
        pose_msg = Pose()
        pose_msg.position.x = pose[0]
        pose_msg.position.y = pose[1]

        q = tf.transformations.quaternion_from_euler(0, 0, pose[2])
        # msg = Quaternion(*q)

        pose_msg.orientation = Quaternion(*q)

        return pose_msg

    def getStartingPose(self):
        # x = (2 * self.random.random() - 1) * 9.5
        # y = self.random.random() * 19 - 9.5
        # t = (2 * self.random.random() - 1) * 3.14
        # pose = [x, y, t]
        # init_pose = self.getPoseMsg(pose=pose)

        return self.init_pose

    def setStartingPose(self):
        L = len(self.maze_barrels)
        l = 0
        while l < L:
            l = 0
            x = (2 * self.random.random() - 1) * 9.5
            y = self.random.random() * 19 - 9.5
            for barrel in self.maze_barrels:
                if np.linalg.norm(np.array([x, y]) - barrel) > 1.:
                    l += 1
        t = (2 * self.random.random() - 1) * 3.14
        pose = [x, y, t]
        self.start_pose = pose
        init_pose = self.getPoseMsg(pose=pose)
        self.init_pose = init_pose

    def getGoal(self):
        # x = (2 * self.random.random() - 1) * 9
        # y = self.random.random() * 18 - 9
        # t = (2 * self.random.random() - 1) * 3.14
        # pose = [x, y, t]
        # init_pose = self.getPoseMsg(pose=pose)
        # pose_stamped = PoseStamped()
        # pose_stamped.pose = init_pose
        # pose_stamped.header.frame_id = "map"
        # return pose_stamped
        return self.goal

    def setGoal(self):
        L = len(self.maze_barrels)
        l = 0
        while l < L:
            l = 0
            while True:
                x = (2 * self.random.random() - 1) * 9
                y = self.random.random() * 18 - 9
                d = np.sqrt((x - self.start_pose[0]) ** 2 + (y - self.start_pose[1]) ** 2)
                if d >= 5.:
                    break
            for barrel in self.maze_barrels:
                if np.linalg.norm(np.array([x, y]) - barrel) > 1.:
                    l += 1
        t = (2 * self.random.random() - 1) * 3.14
        pose = [x, y, t]
        # x, y, t = self.start_pose
        # dx = 5 * np.cos(t)
        # dy = 5 * np.sin(t)
        # if np.abs(x+dx) >= 9.5 or np.abs(y+dy)>=9.5:
        #     t = -t
        #     dx = 5 * np.cos(t)
        #     dy = 5 * np.sin(t)
        # pose = [x + dx, y + dy, t]
        self.goal_pose = pose
        init_pose = self.getPoseMsg(pose=pose)
        pose_stamped = PoseStamped()
        pose_stamped.pose = init_pose
        pose_stamped.header.frame_id = "map"
        self.goal = pose_stamped

    def setupScenario(self):
        self.setStartingPose()
        self.setGoal()
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        self.gazebo_driver.moveObstacles(self.num_barrels, minx=self.minx, maxx=self.maxx, miny=self.miny,
                                         maxy=self.maxy,
                                         grid_spacing=self.min_spacing, start=self.start_pose, goal=self.goal_pose,
                                         d=self.min_spacing, maze=self.maze)
        self.gazebo_driver.unpause()


class MazePredictScenario(MazeScenario):
    def __init__(self, task, gazebo_driver, region_num=5, random=True):
        super(MazePredictScenario, self).__init__(task, gazebo_driver, region_num)
        self.region_barrel = None
        self.random_barrel = random

    def setupScenario(self):
        self.setStartingPose()
        self.setGoal()
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        self.gazebo_driver.moveObstacles(self.num_barrels, minx=self.minx, maxx=self.maxx, miny=self.miny,
                                         maxy=self.maxy,
                                         grid_spacing=self.min_spacing, random=self.random_barrel,
                                         region_num=self.region_num,
                                         start=self.start_pose, goal=self.goal_pose, d=self.min_spacing, maze=self.maze)
        self.density = self.gazebo_driver.density
        self.region_barrel = self.gazebo_driver.region_barrel
        self.gazebo_driver.unpause()


class MazeRLScenario(MazePredictScenario):
    def setupScenario(self):
        self.setStartingPose()
        self.setGoal()
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        self.gazebo_driver.moveObstacles(self.num_barrels, minx=self.minx, maxx=self.maxx, miny=self.miny,
                                         maxy=self.maxy,
                                         grid_spacing=self.min_spacing, random=self.random_barrel,
                                         region_num=self.region_num,
                                         start=self.start_pose, goal=self.goal_pose, d=self.min_spacing, maze=self.maze)
        self.shortest_path()
        self.density = self.gazebo_driver.density
        self.region_barrel = self.gazebo_driver.region_barrel
        self.gazebo_driver.unpause()

    def shortest_path(self):
        # st = time.time()
        barrier = np.array(self.gazebo_driver.barrier)
        multiplier = 16.
        barrier = np.round((barrier + 10.) * multiplier)
        barrier = list(barrier)
        # barriers = set()
        start = tuple(np.round((np.array(self.start_pose)[:2] + 10.) * multiplier))
        goal = tuple(np.round((np.array(self.goal_pose)[:2] + 10.) * multiplier))

        grid = np.zeros((int(2 * 10 * multiplier), int(2 * 10 * multiplier)))
        # from d_star.d_star import DStar
        # pf = DStar(x_start=int(start[0]), y_start=int(start[1]),
        #           x_goal=int(goal[0]), y_goal=int(goal[1]))
        for b in barrier:
            for i in range(4):
                for j in range(4):
                    grid[int(b[0] - i), int(b[1] - j)] = 1
                    grid[int(b[0] - i), int(b[1] + j)] = 1
                    grid[int(b[0] + i), int(b[1] + j)] = 1
                    grid[int(b[0] + i), int(b[1] - j)] = 1

        # graph = AStarGraph(barriers, 200, 200)
        # path, _ = AStarSearch((int(start[0]), int(start[1])), (int(goal[0]), int(goal[1])), graph)
        #pf.replan()
        #path = pf.get_path()

        from heapq import heappop, heappush

        def heuristic(a, b):
            return (b[0] - a[0]) ** 2 + (b[1] - a[1]) ** 2

        def astar(array, start, goal):
            neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]

            close_set = set()
            came_from = {}
            gscore = {start: 0}
            fscore = {start: heuristic(start, goal)}
            oheap = []

            heappush(oheap, (fscore[start], start))

            while oheap:

                current = heappop(oheap)[1]

                if current == goal:
                    data = []
                    while current in came_from:
                        data.append(current)
                        current = came_from[current]
                    return data

                close_set.add(current)
                for i, j in neighbors:
                    neighbor = current[0] + i, current[1] + j
                    tentative_g_score = gscore[current] + heuristic(current, neighbor)
                    if 0 <= neighbor[0] < array.shape[0]:
                        if 0 <= neighbor[1] < array.shape[1]:
                            if array[neighbor[0]][neighbor[1]] == 1:
                                continue
                        else:
                            # array bound y walls
                            continue
                    else:
                        # array bound x walls
                        continue

                    if neighbor in close_set and tentative_g_score >= gscore.get(neighbor, 0):
                        continue

                    if tentative_g_score < gscore.get(neighbor, 0) or neighbor not in [i[1] for i in oheap]:
                        came_from[neighbor] = current
                        gscore[neighbor] = tentative_g_score
                        fscore[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                        heappush(oheap, (fscore[neighbor], neighbor))

            return False

        path = astar(array=grid, start=start, goal=goal)
        print(path)




        self.path = old_div(np.array(path), multiplier) - 10.
        # dt = time.time() - st
        # print(dt)
        # print()


class SectorLaserScenario(SectorScenario):
    def __init__(self, task, gazebo_driver):
        super(SectorLaserScenario, self).__init__(task=task, gazebo_driver=gazebo_driver)

        self.world = "sector_laser"


class FullSectorLaserScenario(SectorLaserScenario):
    def __init__(self, task, gazebo_driver):
        super(FullSectorLaserScenario, self).__init__(task=task, gazebo_driver=gazebo_driver)

        self.world = "full_sector_laser"


class StereoSectorLaserScenario(SectorLaserScenario):
    def __init__(self, task, gazebo_driver):
        super(StereoSectorLaserScenario, self).__init__(task=task, gazebo_driver=gazebo_driver)

        self.world = "stereo_sector_laser"


class SectorExtraScenario(SectorScenario):
    def __init__(self, task, gazebo_driver):
        super(SectorExtraScenario, self).__init__(task=task, gazebo_driver=gazebo_driver)

        self.world = "sector_extra"

        self.num_barrels = task["num_obstacles"] if "num_obstacles" in task else 0

        self.min_spacing = task["min_obstacle_spacing"] if "min_obstacle_spacing" in task else None

        # Zone1 = [[35.5, 14.5], [30, 8.2]]
        # Zone2 = [[25, -10], [19.6, -13.9]]
        # Zone3 = [[-14.4, -13.8], [-9.93, -18.9]]
        # Zone4 = [[-30.5, 10.8], [-24, 7.8]]
        # Zone5 = [[-37.3, 14.8], [-34, 11.1]]
        # Zone6 = [[-33.3, -22.3], [-28.7, -26.5]]
        # Zone7 = [[2.2, 8.4], [8.2, 7]]
        # Zone8 = [[19.5, 24.1], [25.3, 19.2]]
        #
        # zones = [Zone1, Zone2, Zone3, Zone4, Zone5, Zone6, Zone7, Zone8]

        Zone1 = [[-8.5, 9.5], [8.5, -9.5]]
        zones = [Zone1]
        zones = np.swapaxes(zones, 0, 2)

        x1 = zones[0][0]
        x2 = zones[0][1]
        y1 = zones[1][0]
        y2 = zones[1][1]

        self.minx = np.minimum(x1, x2)
        self.maxx = np.maximum(x1, x2)
        self.maxy = np.maximum(y1, y2)
        self.miny = np.minimum(y1, y2)

    @staticmethod
    def getUniqueFieldNames():
        return ["num_obstacles", "seed", "target_id", "init_id", "min_obstacle_spacing"]

    def setupScenario(self):
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        self.gazebo_driver.moveBarrels(self.num_barrels, minx=self.minx, maxx=self.maxx, miny=self.miny, maxy=self.maxy,
                                       grid_spacing=self.min_spacing)
        self.gazebo_driver.unpause()


class FullSectorExtraScenario(SectorExtraScenario):
    def __init__(self, task, gazebo_driver):
        super(FullSectorExtraScenario, self).__init__(task=task, gazebo_driver=gazebo_driver)

        self.world = "full_sector_extra"


class StereoSectorExtraScenario(SectorExtraScenario):
    def __init__(self, task, gazebo_driver):
        super(StereoSectorExtraScenario, self).__init__(task=task, gazebo_driver=gazebo_driver)

        self.world = "stereo_sector_extra"


class FourthFloorScenario(TestingScenario):
    def __init__(self, task, gazebo_driver):
        self.gazebo_driver = gazebo_driver

        self.world = "fourth_floor"

        self.seed = task["seed"] if "seed" in task else 0

        self.init_id = task["init_id"] if "init_id" in task else None
        self.target_id = task[
            "target_id"] if "target_id" in task else None  ##TODO: Replace these with randomly chosen ones
        self.num_barrels = task["num_obstacles"] if "num_obstacles" in task else 0

        self.min_spacing = task["min_obstacle_spacing"] if "min_obstacle_spacing" in task else None

        self.random = random.Random()
        self.random.seed(self.seed)

        self.init_pose = Pose()
        self.init_pose.position.x = -48
        self.init_pose.position.y = 17
        self.init_pose.orientation.x = 0
        self.init_pose.orientation.y = 0
        self.init_pose.orientation.z = 0
        self.init_pose.orientation.w = 1

        self.target_pose = PoseStamped()
        self.target_pose.pose.position.x = 4
        self.target_pose.pose.position.y = 16
        self.target_pose.pose.orientation.x = 0.0
        self.target_pose.pose.orientation.y = 0.0
        self.target_pose.pose.orientation.z = 0.0
        self.target_pose.pose.orientation.w = 1.0
        self.target_pose.header.frame_id = 'map'

        self.target_poses = [[38.87, 11.19, 3.14], [16.05, -15.5, -1.57], [-7.72, -12.5, -1.57], [-17.38, 12.87, -1.57],
                             [-40.77, 14.2, 0], [-33.83, -28.41, 0.785], [-2.34, 13.34, -0.785], [17.44, 25.05, -0.785]]

        if self.init_id is None:
            self.init_id = self.random.randint(0, len(self.target_poses) - 1)
            task["init_id"] = self.init_id

        if self.target_id is None:
            init = self.target_poses[self.init_id]
            dis = []
            for pose in self.target_poses:
                dis.append(math.sqrt((init[0] - pose[0]) ** 2 + (init[1] - pose[1]) ** 2))
            dis_idx = sorted(list(range(len(dis))), key=dis.__getitem__)
            init_rand = self.random.randint(1, 3)
            self.target_id = dis_idx[init_rand]

            task["target_id"] = self.target_id

            # dis = math.sqrt((self.target_poses[self]))

        Zone1 = [[35.5, 14.5], [30, 8.2]]
        Zone2 = [[25, -10], [19.6, -13.9]]
        Zone3 = [[-14.4, -13.8], [-9.93, -18.9]]
        Zone4 = [[-30.5, 10.8], [-24, 7.8]]
        Zone5 = [[-37.3, 14.8], [-34, 11.1]]
        Zone6 = [[-33.3, -22.3], [-28.7, -26.5]]
        Zone7 = [[2.2, 8.4], [8.2, 7]]
        Zone8 = [[19.5, 24.1], [25.3, 19.2]]

        zones = [Zone1, Zone2, Zone3, Zone4, Zone5, Zone6, Zone7, Zone8]

        zones = np.swapaxes(zones, 0, 2)

        x1 = zones[0][0]
        x2 = zones[0][1]
        y1 = zones[1][0]
        y2 = zones[1][1]

        self.minx = np.minimum(x1, x2)
        self.maxx = np.maximum(x1, x2)
        self.maxy = np.maximum(y1, y2)
        self.miny = np.minimum(y1, y2)

    @staticmethod
    def getUniqueFieldNames():
        return ["num_obstacles", "seed", "min_obstacle_spacing"]

    def getPoseMsg(self, pose):
        pose_msg = Pose()
        pose_msg.position.x = pose[0]
        pose_msg.position.y = pose[1]

        q = tf.transformations.quaternion_from_euler(0, 0, pose[2])

        pose_msg.orientation = Quaternion(*q)

        return pose_msg

    def getStartingPose(self):
        pose = self.target_poses[self.init_id]
        init_pose = self.getPoseMsg(pose=pose)

        return init_pose

    def getGoal(self):
        pose = self.target_poses[self.target_id]
        # x, y, t = self.target_poses[self.init_id]
        # dx = 5 * np.cos(t)
        # dy = 5 * np.sin(t)
        # pose = [x + dx, y + dy, t]
        init_pose = self.getPoseMsg(pose=pose)
        pose_stamped = PoseStamped()
        pose_stamped.pose = init_pose
        pose_stamped.header.frame_id = "map"
        return pose_stamped

    def setupScenario(self):
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        self.gazebo_driver.moveBarrels(self.num_barrels, minx=self.minx, maxx=self.maxx, miny=self.miny, maxy=self.maxy,
                                       grid_spacing=self.min_spacing)
        self.gazebo_driver.unpause()


class FourthFloorObstacleScenario(FourthFloorScenario):
    def __init__(self, task, gazebo_driver):
        super(FourthFloorObstacleScenario, self).__init__(task=task, gazebo_driver=gazebo_driver)

        self.world = "fourth_floor_obstacle"

        self.num_obstacles = task["num_obstacles"] if "num_obstacles" in task else 500

    # def getUniqueFieldNames():
    #     return ["num_obstacles", "seed", "min_obstacle_spacing"]

    def setupScenario(self):
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        self.gazebo_driver.moveObstacles(self.num_obstacles, minx=self.minx, maxx=self.maxx, miny=self.miny,
                                         maxy=self.maxy, grid_spacing=self.min_spacing,
                                         start=self.target_poses[self.init_id],
                                         goal=self.target_poses[self.target_id])
        self.gazebo_driver.unpause()


class FourthFloorPredictScenario(FourthFloorObstacleScenario):
    def setupScenario(self):
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        self.gazebo_driver.moveObstacles(self.num_obstacles, minx=self.minx, maxx=self.maxx, miny=self.miny,
                                         maxy=self.maxy, grid_spacing=self.min_spacing,
                                         start=self.target_poses[self.init_id],
                                         goal=self.target_poses[self.target_id],
                                         random=True)  # , model_types=['cylinder', 'square_post', 'pole']
        self.density = self.gazebo_driver.density
        self.gazebo_driver.unpause()


class FullFourthFloorObstacleScenario(FourthFloorObstacleScenario):
    def __init__(self, task, gazebo_driver):
        super(FullFourthFloorObstacleScenario, self).__init__(task=task, gazebo_driver=gazebo_driver)

        self.world = "full_fourth_floor_obstacle"


class StereoFourthFloorObstacleScenario(FourthFloorObstacleScenario):
    def __init__(self, task, gazebo_driver):
        super(StereoFourthFloorObstacleScenario, self).__init__(task=task, gazebo_driver=gazebo_driver)

        self.world = "stereo_fourth_floor_obstacle"


class SparseScenario(TestingScenario):
    def __init__(self, task, gazebo_driver):
        self.gazebo_driver = gazebo_driver

        self.world = "empty_room_20x20"

        self.seed = task["seed"] if "seed" in task else 0
        self.min_obstacle_spacing = task["min_obstacle_spacing"] if "min_obstacle_spacing" in task else 5
        self.num_obstacles = task["num_obstacles"] if "num_obstacles" in task else 500

        self.random = random.Random()
        self.random.seed(self.seed)

        self.minx = [-7]
        self.miny = [-9.5]
        self.maxx = [6.5]
        self.maxy = [9.5]

    @staticmethod
    def getUniqueFieldNames():
        return ["num_obstacles", "seed", "min_obstacle_spacing"]

    def getPoseMsg(self, pose):
        pose_msg = Pose()
        pose_msg.position.x = pose[0]
        pose_msg.position.y = pose[1]

        q = tf.transformations.quaternion_from_euler(0, 0, pose[2])
        # msg = Quaternion(*q)

        pose_msg.orientation = Quaternion(*q)

        return pose_msg

    def getStartingPose(self):
        y = self.random.random() * (18) - 9
        pose = [-9, y, 0]
        init_pose = self.getPoseMsg(pose=pose)

        return init_pose

    def getGoal(self):
        y = self.random.random() * (18) - 9

        pose = [8, y, 0]
        init_pose = self.getPoseMsg(pose=pose)
        pose_stamped = PoseStamped()
        pose_stamped.pose = init_pose
        pose_stamped.header.frame_id = "map"
        return pose_stamped

    def setupScenario(self):
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        self.gazebo_driver.moveObstacles(self.num_obstacles, minx=self.minx, maxx=self.maxx, miny=self.miny,
                                         maxy=self.maxy, grid_spacing=self.min_obstacle_spacing,
                                         model_types=['pole', 'square_post'])
        self.gazebo_driver.unpause()


class DenseScenario(SparseScenario):
    def __init__(self, task, gazebo_driver):
        super(DenseScenario, self).__init__(task=task, gazebo_driver=gazebo_driver)

        self.min_obstacle_spacing = task["min_obstacle_spacing"] if "min_obstacle_spacing" in task else 1


class MediumScenario(SparseScenario):
    def __init__(self, task, gazebo_driver):
        super(MediumScenario, self).__init__(task=task, gazebo_driver=gazebo_driver)

        self.min_obstacle_spacing = task["min_obstacle_spacing"] if "min_obstacle_spacing" in task else 2


class CorridorZigzagScenario(TestingScenario):
    def __init__(self, task, gazebo_driver):
        ##TODO: move common elements (gazebo driver, seed, etc) to super.
        ##TODO: Also make the 'names' of scenarios properties of the scenarios themselves
        # super(Corridor1Scenario, self).__init__(task=task, gazebo_driver=gazebo_driver)
        self.gazebo_driver = gazebo_driver

        self.world = "corridor_zigzag"

        seed = task["seed"] if "seed" in task else 0
        self.min_obstacle_spacing = task["min_obstacle_spacing"]
        self.num_obstacles = task["num_obstacles"]
        self.seed = seed + 1000 * self.num_obstacles  # ensures that changing number of obstacles produces completely different scenes, not just incrementally different

        self.init_pose = Pose()
        self.init_pose.position.x = -6
        self.init_pose.orientation.w = 1

        self.target_pose = PoseStamped()
        self.target_pose.pose.position.x = 6
        self.target_pose.pose.position.y = 3
        self.target_pose.pose.orientation.w = 1.0
        self.target_pose.header.frame_id = 'map'

        Zone1 = [[-4, -1.1], [0, 1.25]]
        Zone2 = [[1.25, -1.1], [3.25, 3.5]]

        zones = [Zone1, Zone2]

        # zones = np.swapaxes(zones, 0, 2)
        mins = np.min(zones, axis=1)
        maxs = np.max(zones, axis=1)
        self.minx = mins[:, 0]
        self.maxx = maxs[:, 0]
        self.maxy = maxs[:, 1]
        self.miny = mins[:, 1]

    def setupScenario(self):
        self.gazebo_driver.checkServicesTopics(10)
        self.gazebo_driver.pause()
        self.gazebo_driver.moveRobot(self.getStartingPose())
        self.gazebo_driver.resetOdom()
        self.gazebo_driver.reset(self.seed)
        self.gazebo_driver.moveObstacles(self.num_obstacles, minx=self.minx, maxx=self.maxx, miny=self.miny,
                                         maxy=self.maxy, grid_spacing=self.min_obstacle_spacing)
        self.gazebo_driver.unpause()


class CorridorZigzagDoorScenario(CorridorZigzagScenario):
    def __init__(self, task, gazebo_driver):
        super(CorridorZigzagDoorScenario, self).__init__(task=task, gazebo_driver=gazebo_driver)
        self.world = "corridor_zigzag_door"
