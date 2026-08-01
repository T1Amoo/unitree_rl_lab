#!/usr/bin/env python3
"""Record a timestamped, wide A1 sim2real trace for MuJoCo comparison."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import socket
import time
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray, String


DEFAULT_OUTPUT_DIR = Path(
    "/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/系统辨识/sim2real/20260801"
)
RIGHT_ALIASES = [
    (f"joint{i}-a1_r", f"joint{i}-r", f"r{i}") for i in range(1, 8)
]
ARRAY_TOPICS = {
    "ball": ("/ball/state", 6),
    "model_action": ("/model_action", 14),
    "raw_action": ("/sim2real/raw_action", 7),
    "raw_q_des": ("/sim2real/raw_q_des", 7),
    "q_des": ("/sim2real/q_des", 7),
    "dq_des": ("/sim2real/dq_des", 7),
    "timing": ("/sim2real/timing", 9),
    "frame": ("/sim2real/frame", 39),
    "obs": ("/sim2real/obs", 195),
}
GATE_FIELDS = {
    "gate_engaged": r"engaged=([^ ]+)",
    "gate_live": r"live=([^ ]+)",
    "gate_reason": r"reason=([^ ]+)",
    "gate_first_bounce_x": r"first_bounce_x=([^ ]+)",
    "gate_own_bounces": r"own_bounces=([^ ]+)",
    "gate_ball_stale": r"ball_stale=([^ ]+)",
    "gate_pred": r"pred=\[([^]]+)\]",
}


class Latest:
    def __init__(self) -> None:
        self.seq = 0
        self.recv_epoch_ns = 0
        self.recv_mono_ns = 0
        self.value: dict[str, object] = {}

    def update(self, value: dict[str, object]) -> None:
        self.seq += 1
        self.recv_epoch_ns = time.time_ns()
        self.recv_mono_ns = time.monotonic_ns()
        self.value = value


def _ordered(values: list[float], names: list[str]) -> list[float]:
    if not values:
        return [float("nan")] * 7
    if not names and len(values) >= 7:
        return list(values[:7])
    by_name = {name: value for name, value in zip(names, values)}
    out = []
    for aliases in RIGHT_ALIASES:
        out.append(next((float(by_name[a]) for a in aliases if a in by_name), float("nan")))
    return out


class Sim2RealTraceRecorder(Node):
    def __init__(self, csv_path: Path, rate_hz: float) -> None:
        super().__init__("a1_sim2real_trace_recorder")
        self.csv_path = csv_path
        self.rate_hz = rate_hz
        self.rows = 0
        self.latest: dict[str, Latest] = {
            name: Latest()
            for name in [
                "camera", "joint", "fsm", "gate", "policy_enable", "motor_enable", *ARRAY_TOPICS
            ]
        }

        self.create_subscription(PoseStamped, "/pingpong_location", self._camera_cb, qos_profile_sensor_data)
        self.create_subscription(JointState, "/right_joint_states", self._joint_cb, qos_profile_sensor_data)
        self.create_subscription(String, "/a1_tt/fsm_state", lambda m: self._text_cb("fsm", m), 10)
        self.create_subscription(String, "/sim2real/gate", lambda m: self._text_cb("gate", m), 10)
        self.create_subscription(Bool, "/a1_tt/policy_enable", lambda m: self._bool_cb("policy_enable", m), 10)
        self.create_subscription(Bool, "/model_control/enable", lambda m: self._bool_cb("motor_enable", m), 10)
        for name, (topic, _) in ARRAY_TOPICS.items():
            self.create_subscription(
                Float64MultiArray,
                topic,
                lambda msg, slot=name: self._array_cb(slot, msg),
                10,
            )

        self.fieldnames = self._fieldnames()
        self.csv_file = csv_path.open("w", newline="")
        self.writer = csv.DictWriter(self.csv_file, fieldnames=self.fieldnames)
        self.writer.writeheader()
        self.timer = self.create_timer(1.0 / rate_hz, self._sample)
        self.get_logger().info(f"recording {rate_hz:.1f} Hz trace -> {csv_path}")

    @staticmethod
    def _fieldnames() -> list[str]:
        fields = ["sample_epoch_ns", "sample_mono_ns", "sample_index"]
        fields += [
            "camera_seq", "camera_recv_epoch_ns", "camera_recv_age_ms", "camera_source_ns",
            "camera_source_age_ms", "camera_frame", "camera_x", "camera_y", "camera_z",
        ]
        fields += ["ball_seq", "ball_recv_epoch_ns", "ball_recv_age_ms"]
        fields += [f"ball_{name}" for name in ("x", "y", "z", "vx", "vy", "vz")]
        fields += ["joint_seq", "joint_recv_epoch_ns", "joint_recv_age_ms", "joint_source_ns"]
        for prefix in ("q", "dq", "effort"):
            fields += [f"actual_{prefix}_{i}" for i in range(1, 8)]
        for name in ("fsm", "gate", "policy_enable", "motor_enable"):
            fields += [f"{name}_seq", f"{name}_recv_epoch_ns", f"{name}_recv_age_ms", name]
        fields += list(GATE_FIELDS)
        fields += ["gate_pred_x", "gate_pred_y", "gate_pred_z"]
        for name, (_, width) in ARRAY_TOPICS.items():
            fields += [f"{name}_seq", f"{name}_recv_epoch_ns", f"{name}_recv_age_ms"]
            fields += [f"{name}_{i}" for i in range(width)]
        return fields

    def _camera_cb(self, msg: PoseStamped) -> None:
        source_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.latest["camera"].update(
            {
                "source_ns": source_ns,
                "frame": msg.header.frame_id,
                "xyz": [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            }
        )

    def _joint_cb(self, msg: JointState) -> None:
        source_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        names = list(msg.name)
        self.latest["joint"].update(
            {
                "source_ns": source_ns,
                "q": _ordered(list(msg.position), names),
                "dq": _ordered(list(msg.velocity), names),
                "effort": _ordered(list(msg.effort), names),
            }
        )

    def _text_cb(self, name: str, msg: String) -> None:
        self.latest[name].update({"data": msg.data})

    def _bool_cb(self, name: str, msg: Bool) -> None:
        self.latest[name].update({"data": int(msg.data)})

    def _array_cb(self, name: str, msg: Float64MultiArray) -> None:
        self.latest[name].update({"data": list(msg.data)})

    @staticmethod
    def _slot_base(row: dict[str, object], name: str, slot: Latest, now_ns: int) -> None:
        row[f"{name}_seq"] = slot.seq
        row[f"{name}_recv_epoch_ns"] = slot.recv_epoch_ns or ""
        row[f"{name}_recv_age_ms"] = (now_ns - slot.recv_epoch_ns) / 1e6 if slot.recv_epoch_ns else ""

    def _sample(self) -> None:
        now_ns = time.time_ns()
        row: dict[str, object] = {key: "" for key in self.fieldnames}
        row.update(sample_epoch_ns=now_ns, sample_mono_ns=time.monotonic_ns(), sample_index=self.rows)

        cam = self.latest["camera"]
        self._slot_base(row, "camera", cam, now_ns)
        if cam.value:
            source_ns = int(cam.value["source_ns"])
            row.update(
                camera_source_ns=source_ns,
                camera_source_age_ms=(now_ns - source_ns) / 1e6 if source_ns else "",
                camera_frame=cam.value["frame"],
            )
            for key, value in zip(("camera_x", "camera_y", "camera_z"), cam.value["xyz"]):
                row[key] = value

        ball = self.latest["ball"]
        self._slot_base(row, "ball", ball, now_ns)
        for key, value in zip(
            [f"ball_{name}" for name in ("x", "y", "z", "vx", "vy", "vz")],
            ball.value.get("data", []),
        ):
            row[key] = value

        joint = self.latest["joint"]
        self._slot_base(row, "joint", joint, now_ns)
        if joint.value:
            row["joint_source_ns"] = joint.value["source_ns"]
            for prefix in ("q", "dq", "effort"):
                for i, value in enumerate(joint.value[prefix], 1):
                    row[f"actual_{prefix}_{i}"] = value

        for name in ("fsm", "gate", "policy_enable", "motor_enable"):
            slot = self.latest[name]
            self._slot_base(row, name, slot, now_ns)
            row[name] = slot.value.get("data", "")

        gate_text = str(self.latest["gate"].value.get("data", ""))
        for key, pattern in GATE_FIELDS.items():
            match = re.search(pattern, gate_text)
            if match:
                row[key] = match.group(1)
        if row["gate_pred"]:
            for key, value in zip(
                ("gate_pred_x", "gate_pred_y", "gate_pred_z"),
                str(row["gate_pred"]).split(","),
            ):
                row[key] = value.strip()

        for name, (_, width) in ARRAY_TOPICS.items():
            slot = self.latest[name]
            self._slot_base(row, name, slot, now_ns)
            for i, value in enumerate(slot.value.get("data", [])[:width]):
                row[f"{name}_{i}"] = value

        self.writer.writerow(row)
        self.rows += 1
        if self.rows % 100 == 0:
            self.csv_file.flush()

    def close(self) -> None:
        if not self.csv_file.closed:
            self.csv_file.flush()
            self.csv_file.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--label", default="a1_backhand_9700")
    parser.add_argument("--rate-hz", type=float, default=100.0)
    args, ros_args = parser.parse_known_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = f"{args.label}_{stamp}"
    csv_path = args.output_dir / f"{stem}.csv"
    metadata_path = args.output_dir / f"{stem}.metadata.json"
    metadata = {
        "start_epoch_ns": time.time_ns(),
        "start_local": datetime.now().astimezone().isoformat(),
        "hostname": socket.gethostname(),
        "rate_hz": args.rate_hz,
        "csv": str(csv_path),
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", ""),
        "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION", ""),
        "cyclonedds_uri": os.environ.get("CYCLONEDDS_URI", ""),
        "topics": {name: topic for name, (topic, _) in ARRAY_TOPICS.items()},
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    rclpy.init(args=ros_args)
    node = Sim2RealTraceRecorder(csv_path, args.rate_hz)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        metadata.update(
            end_epoch_ns=time.time_ns(),
            end_local=datetime.now().astimezone().isoformat(),
            rows=node.rows,
        )
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")
        node.destroy_node()
        # SIGINT handled by rclpy may already have shut the default context down.
        if rclpy.ok():
            rclpy.shutdown()
        print(f"trace_csv={csv_path}")
        print(f"metadata={metadata_path}")


if __name__ == "__main__":
    main()
