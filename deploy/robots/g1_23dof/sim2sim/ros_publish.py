"""Publishes ball + base positions as ROS2 PoseStamped, mirroring the real
VRPN mocap interface (topic names, BEST_EFFORT QoS, and the room frame M).
mjData positions are in training-world W; we emit p_M = p_W - input_frame.origin
so the C++ deploy's p_W = R*p_M + origin reconstructs W identically -> the same
config.yaml drives both sim2sim and sim2real. See the block comment below.
"""
import os
import random

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from geometry_msgs.msg import PoseStamped

# ---------------------------------------------------------------------------
# Make the sim mocap output identical to the real VRPN mocap so ONE config.yaml
# serves both sim2sim and sim2real:
#   - topic names match config.yaml TableTennis.ros (vrpn rigid-body topics)
#   - BEST_EFFORT QoS (qos_profile_sensor_data) matches the real VRPN publisher
#     and the C++ subscriber (SensorDataQoS).
#   - frame: the real mocap reports in its room frame M (origin at the table-top
#     center); the C++ maps p_W = R*p_M + origin_in_training_world. mjData gives
#     positions already in training-world W, so to emit M we send p_M = p_W -
#     origin (rotation is identity for this table-aligned setup). The controller
#     adds origin back and recovers the exact sim-W values -> sim2sim is a net
#     pass-through, AND the same origin=[0,0,0.76] real calibration applies.
#     The (0,0,0) no-ball sentinel becomes (0,0,-origin_z) on the wire and maps
#     back to (0,0,0) in W (norm < 1e-6 -> C++ have_ball=false), so it survives.
# ---------------------------------------------------------------------------
BALL_TOPIC = "/vrpn_mocap/U_Tracker0/pose"
BASE_TOPIC = "/vrpn_mocap/g1/pose"


def _load_origin():
    """Read TableTennis.input_frame.origin_in_training_world from config.yaml so
    the publisher and the C++ deploy share a single source of truth. Falls back
    to the calibrated table-top height if the file/key is missing."""
    import yaml
    cfg_path = os.path.join(os.path.dirname(__file__), "..", "config", "config.yaml")
    try:
        with open(cfg_path) as f:
            cfg = yaml.safe_load(f)
        o = cfg["FSM"]["TableTennis"]["input_frame"]["origin_in_training_world"]
        return [float(o[0]), float(o[1]), float(o[2])]
    except Exception as e:  # noqa: BLE001
        print(f"[ros_publish] could not read input_frame origin ({e}); using [0,0,0.76]", flush=True)
        return [0.0, 0.0, 0.76]

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
        # BEST_EFFORT keep_last(10) — matches real VRPN + the C++ SensorDataQoS sub.
        self.ball_pub = self.create_publisher(PoseStamped, BALL_TOPIC, qos_profile_sensor_data)
        self.base_pub = self.create_publisher(PoseStamped, BASE_TOPIC, qos_profile_sensor_data)
        # origin to subtract so we emit in the real mocap room frame M (see header).
        self._origin = _load_origin()
        print(f"[ros_publish] ball={BALL_TOPIC} base={BASE_TOPIC} "
              f"frame_origin_subtracted={self._origin} QoS=BEST_EFFORT", flush=True)

    def _to_M(self, pos):
        """W -> M: subtract origin (rotation is identity for this setup)."""
        return [pos[0] - self._origin[0], pos[1] - self._origin[1], pos[2] - self._origin[2]]

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
            self.ball_pub.publish(self._msg(self._to_M(bp)))
        if not _base_dropped():
            self.base_pub.publish(self._msg(self._to_M(list(base_pos)), base_quat))
