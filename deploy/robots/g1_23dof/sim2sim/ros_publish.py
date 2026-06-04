"""Publishes ball + base positions as ROS2 PoseStamped, mirroring the real
Nokov mocap interface. Frame = table frame (table at world origin in the MJCF),
so values are world ball/pelvis positions straight from mjData.
"""
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped


class MocapPublisher(Node):
    def __init__(self):
        super().__init__("tt_mocap_publisher")
        self.ball_pub = self.create_publisher(PoseStamped, "/mocap/ball/pose", 10)
        self.base_pub = self.create_publisher(PoseStamped, "/mocap/base/pose", 10)

    def _msg(self, pos, quat=(1.0, 0.0, 0.0, 0.0)):
        m = PoseStamped()
        m.header.stamp = self.get_clock().now().to_msg()
        m.header.frame_id = "table"
        m.pose.position.x = float(pos[0])
        m.pose.position.y = float(pos[1])
        m.pose.position.z = float(pos[2])
        m.pose.orientation.w = float(quat[0])
        m.pose.orientation.x = float(quat[1])
        m.pose.orientation.y = float(quat[2])
        m.pose.orientation.z = float(quat[3])
        return m

    def publish(self, ball_pos, base_pos, base_quat):
        self.ball_pub.publish(self._msg(ball_pos))
        self.base_pub.publish(self._msg(base_pos, base_quat))
