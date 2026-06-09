"""Publishes ball + base positions as ROS2 PoseStamped, mirroring the real
Nokov mocap interface. Frame = table frame (table at world origin in the MJCF),
so values are world ball/pelvis positions straight from mjData.
"""
import os
import random

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped

# ---------------------------------------------------------------------------
# Optional mocap fault injection — OFF by default.
# Set TT_FAULT to a comma/substring list of: drop, jitter, false, baseocc
#   drop    – ~0.5 s ball dropout every 2 s (25-frame windows)
#   jitter  – Gaussian noise (σ=0.01 m) on every ball position
#   false   – spurious false-positive ball detections (not implemented here;
#             PoseStamped is a 1-point message so a second candidate cannot be
#             injected cleanly without changing the topic type; skip for now)
#   baseocc – ~0.5 s base-pose dropout every 2.5 s
# ---------------------------------------------------------------------------
_TT_FAULT = os.environ.get("TT_FAULT", "")   # comma/substring list: drop,jitter,false,baseocc
_fault_n = {"ball": 0, "base": 0}


def _ball_fault(pos):
    """Return possibly-perturbed ball pos, or None to simulate a dropout (skip publish)."""
    if not _TT_FAULT:
        return pos
    _fault_n["ball"] += 1
    n = _fault_n["ball"]
    if "drop" in _TT_FAULT and (n // 25) % 4 == 0:        # ~0.5 s dropout every 2 s
        return None
    if "jitter" in _TT_FAULT:
        pos = [pos[0] + random.gauss(0, 0.01),
               pos[1] + random.gauss(0, 0.01),
               pos[2] + random.gauss(0, 0.01)]
    return pos


def _extra_false_ball():
    """When 'false' fault active, occasionally return a spurious reflective point (else None)."""
    if "false" in _TT_FAULT and random.random() < 0.3:
        return [random.uniform(-1.2, 1.2), random.uniform(-0.8, 0.8), random.uniform(0.8, 1.6)]
    return None


def _base_dropped():
    """When 'baseocc' fault active, drop the base pose in ~0.5 s windows every 2.5 s."""
    if "baseocc" not in _TT_FAULT:
        return False
    _fault_n["base"] += 1
    return (_fault_n["base"] // 25) % 5 == 0


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
        bp = _ball_fault(list(ball_pos))
        if bp is not None:
            self.ball_pub.publish(self._msg(bp))
        if not _base_dropped():
            self.base_pub.publish(self._msg(base_pos, base_quat))
