import rosbag
from geometry_msgs.msg import Pose, PoseStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan, PointCloud, Image
import rospy
import numpy as np
from skimage.transform import rescale
from std_msgs.msg import String, Float32, Float64, Time, Float64MultiArray
from cv_bridge import CvBridge


class ResultRecorders:
    def __init__(self, filename):
        if not filename[-4:] is '.bag':
            filename += '.bag'
        self.bag = rosbag.Bag(filename, 'w')

    def get_recorders(self, recorder):
        # for t in topics:
        topic = recorder['topic']
        if topic == 'result' or topic == 'string':
            return StringRecorder(bag=self.bag, key=topic)
        elif topic == 'time':
            return TimeRecorder(bag=self.bag, key=topic)
        elif topic == 'float64':
            return Float64Recorder(bag=self.bag, key=topic)
        elif topic == 'float32' \
                or topic == 'params' \
                or topic == 'path_length' \
                or topic == 'predict' \
                or topic == 'accuracy' \
                or topic == 'reward' \
                or topic == 'max_global_plan_lookahead_dist':
            return Float32Recorder(bag=self.bag, key=topic)
        elif topic == 'laser_scan':
            node = recorder['node']
            return LaserScanRecorder(bag=self.bag, node=node, key=topic)
        elif topic == 'depth':
            node = recorder['node']
            return DepthRecorder(bag=self.bag, node=node, key=topic)
        elif topic == 'point_cloud':
            node = recorder['node']
            return PointCloudRecorder(bag=self.bag, node=node, key=topic)

    def close(self):
        self.bag.close()


class ResultRecorder(object):
    def __init__(self, bag, key):
        self.bag = bag
        self.key = key

    def start(self):
        pass


class StringRecorder(ResultRecorder):
    def __init__(self, bag, key):
        super(StringRecorder, self).__init__(bag, key)

    def write(self, data):
        if isinstance(data, dict):
            for key in data.keys():
                entry = String()
                entry.data = data[key]
                self.bag.write(self.key + '_' + str(key), entry)
        else:
            entry = String()
            entry.data = data
            self.bag.write(self.key, entry)


class TimeRecorder(ResultRecorder):
    def __init__(self, bag, key):
        super(TimeRecorder, self).__init__(bag, key)

    def write(self, data):
        if isinstance(data, dict):
            for key in data.keys():
                entry = Time()
                entry.data = data[key]
                self.bag.write(self.key + '_' + str(key), entry)
        else:
            entry = Time()
            entry.data = data
            self.bag.write(self.key, entry)


class Float32Recorder(ResultRecorder):
    def __init__(self, bag, key):
        super(Float32Recorder, self).__init__(bag, key)

    def write(self, data):
        if isinstance(data, dict):
            for key in data.keys():
                entry = Float32()
                entry.data = data[key]
                self.bag.write(self.key + '_' + str(key), entry)
        else:
            entry = Float32()
            entry.data = data
            self.bag.write(self.key, entry)


class Float64Recorder(ResultRecorder):
    def __init__(self, bag, key):
        super(Float64Recorder, self).__init__(bag, key)

    def write(self, data):
        if isinstance(data, dict):
            for key in data.keys():
                entry = Float64()
                entry.data = data[key]
                self.bag.write(self.key + '_' + str(key), entry)
        else:
            entry = Float64()
            entry.data = data
            self.bag.write(self.key, entry)


class ArrayRecorder(ResultRecorder):
    def __init__(self, bag, key):
        super(ArrayRecorder, self).__init__(bag, key)

    def write(self, data):
        if isinstance(data, dict):
            for key in data.keys():
                entry = Float64MultiArray()
                entry.data = data[key]
                self.bag.write(self.key + '_' + str(key), entry)
        else:
            entry = Float64MultiArray()
            entry.data = data
            self.bag.write(self.key, entry)


class DepthRecorder(ResultRecorder):
    def __init__(self, bag, node, key, interval=5):
        super(DepthRecorder, self).__init__(bag, key)
        self.node = node
        self.interval = interval
        self.counter = 1
        self.bridge = CvBridge()

    def start(self):
        self.sub = rospy.Subscriber(self.node, Image, self.DepthCB)

    def DepthCB(self, data):
        if self.counter % self.interval == 0:
            image = self.bridge.imgmsg_to_cv2(data, "passthrough")
            image_rescaled = np.array(rescale(image, 0.5), dtype=np.float32)
            new_data = self.bridge.cv2_to_imgmsg(image_rescaled, 'passthrough')
            data.data = new_data.data
            self.bag.write(self.key, data)
        self.counter += 1

    def close(self):
        self.sub.unregister()


class LaserScanRecorder(ResultRecorder):
    def __init__(self, bag, node, key, interval=5):
        super(LaserScanRecorder, self).__init__(bag, key)
        self.node = node
        self.interval = interval
        self.counter = 0

    def start(self):
        self.sub = rospy.Subscriber(self.node, LaserScan, self.LaserScanCB)

    def LaserScanCB(self, data):
        if self.counter % self.interval == 0:
            self.bag.write(self.key, data)
        self.counter += 1

    def close(self):
        self.sub.unregister()


class PointCloudRecorder(ResultRecorder):
    def __init__(self, bag, node, key):
        super(PointCloudRecorder, self).__init__(bag, key)
        self.node = node

    def start(self):
        self.sub = rospy.Subscriber(self.node, PointCloud, self.PointCloudCB)

    def PointCloudCB(self, data):
        self.bag.write(self.key, data)

    def close(self):
        self.sub.unregister()


class OdometryRecorder(ResultRecorder):
    def __init__(self, bag, node, key):
        super(OdometryRecorder, self).__init__(bag, key)
        self.node = node

    def start(self):
        self.sub = rospy.Subscriber(self.node, Odometry, self.OdometryCB)

    def OdometryCB(self, data):
        self.bag.write(self.key, data)

    def close(self):
        self.sub.unregister()


class PoseRecorder(ResultRecorder):
    def __init__(self, bag, node, key):
        super(PoseRecorder, self).__init__(bag, key)
        self.node = node

    def start(self):
        self.sub = rospy.Subscriber(self.node, Pose, self.PoseCB)

    def PoseCB(self, data):
        pose_stamped = PoseStamped()
        pose_stamped.pose = data
        pose_stamped.header.frame_id = "map"
        self.bag.write(self.key, pose_stamped)

    def close(self):
        self.sub.unregister()
