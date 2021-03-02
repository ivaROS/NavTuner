import rosbag
import rospy

from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped
from rosgraph_msgs.msg import Clock

path0 = Path()
path1 = Path()
path2 = Path()
rospy.init_node('path_node')
pub0 = rospy.Publisher('/path0', Path, queue_size=100)
pub1 = rospy.Publisher('/path1', Path, queue_size=100)
pub2 = rospy.Publisher('/path2', Path, queue_size=100)

bag0 = rosbag.Bag('planner_freq_0.0625.bag', 'r')
for _, msg, _ in bag0.read_messages(topics=['pose']):
    msg.header.frame_id = 'map'
    path0.header = msg.header
    # pose = PoseStamped()
    # pose.header = msg.header
    # pose.pose = msg.pose.pose
    path0.poses.append(msg)
bag0.close()
# pub0.publish(path0)

bag1 = rosbag.Bag('planner_freq_0.25.bag', 'r')
for _, msg, _ in bag1.read_messages(topics=['pose']):
    msg.header.frame_id = 'map'
    path1.header = msg.header
    # pose = PoseStamped()
    # pose.header = msg.header
    # pose.pose = msg.pose.pose
    path1.poses.append(msg)
bag1.close()
# pub1.publish(path1)

bag2 = rosbag.Bag('planner_freq_0.125.bag', 'r')
for _, msg, _ in bag2.read_messages(topics=['pose']):
    msg.header.frame_id = 'map'
    path2.header = msg.header
    # pose = PoseStamped()
    # pose.header = msg.header
    # pose.pose = msg.pose.pose
    path2.poses.append(msg)
bag2.close()
# pub2.publish(path2)

def clockCB(data):
    pub0.publish(path0)
    pub1.publish(path1)
    pub2.publish(path2)

# sub = rospy.Subscriber('/clock', Clock, clockCB)


if __name__ == '__main__':
    while True:
        pub0.publish(path0)
        pub1.publish(path1)
        pub2.publish(path2)
    # rospy.spin()