"""ROS2 node: A1 table-tennis policy -> armcontrol /model_action."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String

from .policy_runtime import (
    DEFAULT_POLICY,
    DEFAULT_RIGHT_Q,
    BallGateConfig,
    BallValidityGate,
    A1DeployPolicy,
    ordered_joint_vector,
)

BRIDGE_MAX_DELTA_PER_TICK = [0.050, 0.050, 0.050, 0.100, 0.100, 0.100, 0.100]


def _bool_param(node: Node, name: str, default: bool) -> bool:
    value = node.declare_parameter(name, default).value
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _double_array_param(node: Node, name: str, default: list[float]) -> list[float]:
    value = node.declare_parameter(name, default).value
    return [float(x) for x in value]


class A1PolicyBridge(Node):
    def __init__(self) -> None:
        super().__init__("a1_policy_bridge")

        policy_path_value = str(self.declare_parameter("policy_path", str(DEFAULT_POLICY)).value)
        self.policy_path = Path(policy_path_value) if policy_path_value else DEFAULT_POLICY
        predictor_path_value = str(self.declare_parameter("predictor_path", "").value)
        self.predictor_path = Path(predictor_path_value) if predictor_path_value else None
        self.use_predictor = _bool_param(self, "use_predictor", True)
        self.control_hz = float(self.declare_parameter("control_hz", 50.0).value)
        self.joint_timeout_s = float(self.declare_parameter("joint_timeout_s", 0.25).value)
        self.ball_timeout_s = float(self.declare_parameter("ball_timeout_s", 0.20).value)
        self.publish_actions = _bool_param(self, "publish_actions", True)
        self.publish_position_velocity = _bool_param(self, "publish_position_velocity", False)
        self.enable_on_start = _bool_param(self, "enable_on_start", False)
        self.disable_on_stale_joint = _bool_param(self, "disable_on_stale_joint", True)
        self.hold_when_ball_stale = _bool_param(self, "hold_when_ball_stale", False)
        self.diag_every = int(self.declare_parameter("diag_every", 50).value)
        self.action_topic = str(self.declare_parameter("action_topic", "/model_action").value)
        self.enable_topic = str(self.declare_parameter("enable_topic", "/model_control/enable").value)
        self.joint_state_topic = str(self.declare_parameter("joint_state_topic", "/right_joint_states").value)
        self.ball_state_topic = str(self.declare_parameter("ball_state_topic", "/ball/state").value)
        self.max_delta_per_tick = np.asarray(
            _double_array_param(self, "max_delta_per_tick", BRIDGE_MAX_DELTA_PER_TICK),
            dtype=np.float64,
        )
        if self.max_delta_per_tick.shape != (7,):
            raise ValueError("max_delta_per_tick must contain 7 values")

        gate_cfg = BallGateConfig(
            confirm_frames=int(self.declare_parameter("gate_confirm_frames", 1).value),
            coast_frames=int(self.declare_parameter("gate_coast_frames", 1).value),
        )
        self.gate = BallValidityGate(gate_cfg)
        self.policy = A1DeployPolicy(self.policy_path, self.predictor_path, self.use_predictor)

        self.q: Optional[np.ndarray] = None
        self.dq = np.zeros(7, dtype=np.float64)
        self.last_joint_time = None
        self.ball_pos: Optional[np.ndarray] = None
        self.ball_vel = np.zeros(3, dtype=np.float64)
        self.last_ball_time = None
        self.prev_ball_pos: Optional[np.ndarray] = None
        self.prev_ball_time = None
        self.last_pub_q: Optional[np.ndarray] = None
        self.tick = 0
        self.enabled_sent = False

        self.action_pub = self.create_publisher(Float64MultiArray, self.action_topic, 10)
        self.enable_pub = self.create_publisher(Bool, self.enable_topic, 10)
        self.raw_action_pub = self.create_publisher(Float64MultiArray, "/sim2real/raw_action", 10)
        self.q_des_pub = self.create_publisher(Float64MultiArray, "/sim2real/q_des", 10)
        self.gate_pub = self.create_publisher(String, "/sim2real/gate", 10)

        self.create_subscription(JointState, self.joint_state_topic, self.joint_cb, 10)
        self.create_subscription(Float64MultiArray, self.ball_state_topic, self.ball_cb, 10)
        period = 1.0 / max(self.control_hz, 1e-6)
        self.timer = self.create_timer(period, self.control_tick)

        self.publish_enable(self.enable_on_start)
        self.get_logger().info(
            "a1_policy_bridge ready: policy=%s joint_topic=%s ball_topic=%s action_topic=%s publish_actions=%s enable_on_start=%s",
            str(self.policy_path),
            self.joint_state_topic,
            self.ball_state_topic,
            self.action_topic,
            self.publish_actions,
            self.enable_on_start,
        )
        self.get_logger().info(
            "max_delta_per_tick=%s",
            np.round(self.max_delta_per_tick, 3).tolist(),
        )

    def now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def publish_enable(self, enabled: bool) -> None:
        msg = Bool()
        msg.data = bool(enabled)
        self.enable_pub.publish(msg)
        self.enabled_sent = bool(enabled)

    def joint_cb(self, msg: JointState) -> None:
        q = ordered_joint_vector(msg.name, msg.position, self.q if self.q is not None else DEFAULT_RIGHT_Q)
        if q is None:
            self.get_logger().warn("joint state did not contain 7 usable right-arm positions")
            return
        dq = ordered_joint_vector(msg.name, msg.velocity, self.dq)
        self.q = q
        self.dq = np.zeros(7, dtype=np.float64) if dq is None else dq
        self.last_joint_time = self.now_sec()
        if self.last_pub_q is None:
            self.last_pub_q = q.copy()
            self.policy.reset(q)

    def ball_cb(self, msg: Float64MultiArray) -> None:
        if len(msg.data) < 3:
            self.get_logger().warn("ball_state expects data=[x,y,z,vx,vy,vz] or [x,y,z]")
            return
        now = self.now_sec()
        pos = np.array(msg.data[:3], dtype=np.float64)
        if len(msg.data) >= 6:
            vel = np.array(msg.data[3:6], dtype=np.float64)
        elif self.prev_ball_pos is not None and self.prev_ball_time is not None:
            dt = max(now - self.prev_ball_time, 1e-6)
            vel = (pos - self.prev_ball_pos) / dt
        else:
            vel = np.zeros(3, dtype=np.float64)
        self.prev_ball_pos = pos.copy()
        self.prev_ball_time = now
        self.ball_pos = pos
        self.ball_vel = vel
        self.last_ball_time = now

    def stale(self, stamp: Optional[float], timeout_s: float) -> bool:
        return stamp is None or self.now_sec() - stamp > timeout_s

    def clamp_delta(self, q_des: np.ndarray) -> np.ndarray:
        if self.last_pub_q is None or np.all(self.max_delta_per_tick <= 0.0):
            return q_des
        delta = q_des - self.last_pub_q
        clipped = np.clip(delta, -self.max_delta_per_tick, self.max_delta_per_tick)
        return self.last_pub_q + clipped

    def control_tick(self) -> None:
        self.tick += 1
        if self.q is None:
            return
        if self.stale(self.last_joint_time, self.joint_timeout_s):
            if self.disable_on_stale_joint and self.enabled_sent:
                self.publish_enable(False)
            self.get_logger().warn("stale joint state; suppressing policy action", throttle_duration_sec=1.0)
            return

        ball_stale = self.stale(self.last_ball_time, self.ball_timeout_s)
        if ball_stale:
            gate_out = self.gate.update(np.full(3, np.nan), np.full(3, np.nan))
            if self.hold_when_ball_stale:
                self.policy.reset(self.q)
                q_des = self.q.copy()
                raw_action = np.zeros(7, dtype=np.float32)
                ball_pred = np.full(3, np.nan, dtype=np.float32)
            else:
                step = self.policy.step(self.q, self.dq, np.zeros(3), np.zeros(3), False, 1.0 / self.control_hz)
                q_des, raw_action, ball_pred = step.q_des, step.raw_action, step.ball_pred
        else:
            assert self.ball_pos is not None
            gate_out = self.gate.update(self.ball_pos, self.ball_vel)
            step = self.policy.step(self.q, self.dq, self.ball_pos, self.ball_vel, gate_out.engaged, 1.0 / self.control_hz)
            q_des, raw_action, ball_pred = step.q_des, step.raw_action, step.ball_pred

        q_des = self.clamp_delta(np.asarray(q_des, dtype=np.float64).reshape(7))
        dq_des = (q_des - (self.last_pub_q if self.last_pub_q is not None else self.q)) * self.control_hz
        self.last_pub_q = q_des.copy()

        if self.publish_actions:
            action_msg = Float64MultiArray()
            if self.publish_position_velocity:
                action_msg.data = q_des.tolist() + dq_des.tolist()
            else:
                action_msg.data = q_des.tolist()
            self.action_pub.publish(action_msg)

        raw_msg = Float64MultiArray()
        raw_msg.data = [float(x) for x in raw_action]
        self.raw_action_pub.publish(raw_msg)

        q_msg = Float64MultiArray()
        q_msg.data = [float(x) for x in q_des]
        self.q_des_pub.publish(q_msg)

        gate_msg = String()
        gate_msg.data = (
            f"engaged={int(gate_out.engaged)} live={int(gate_out.live)} reason={gate_out.reason} "
            f"first_bounce_x={gate_out.first_bounce_x:.3f} ball_stale={int(ball_stale)} "
            f"pred={np.round(ball_pred, 3).tolist()}"
        )
        self.gate_pub.publish(gate_msg)

        if self.diag_every > 0 and self.tick % self.diag_every == 0:
            max_action = float(np.max(np.abs(raw_action))) if np.all(np.isfinite(raw_action)) else math.nan
            self.get_logger().info(
                "tick=%d gate=%s/%s q_des=%s raw_max=%.2f pred=%s",
                self.tick,
                int(gate_out.engaged),
                gate_out.reason,
                np.round(q_des, 3).tolist(),
                max_action,
                np.round(ball_pred, 3).tolist(),
            )


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = A1PolicyBridge()
    try:
        rclpy.spin(node)
    finally:
        node.publish_enable(False)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
