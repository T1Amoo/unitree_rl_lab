#!/usr/bin/env python3
"""Record a timestamped, wide A1 sim2real trace for MuJoCo comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import socket
import time
from datetime import datetime
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import BatteryState, JointState
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


def _file_identity(path_text: str) -> dict[str, object]:
    """Return a reproducible deployment-artifact identity without requiring it."""
    if not path_text:
        return {"path": "", "available": False, "reason": "not_provided"}
    path = Path(path_text).expanduser().resolve()
    identity: dict[str, object] = {"path": str(path), "available": path.is_file()}
    if not path.is_file():
        identity["reason"] = "not_a_readable_file"
        return identity
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    stat = path.stat()
    identity.update(
        sha256=digest.hexdigest(),
        size_bytes=stat.st_size,
        mtime_epoch_ns=stat.st_mtime_ns,
    )
    return identity


class Sim2RealTraceRecorder(Node):
    def __init__(
        self,
        csv_path: Path,
        rate_hz: float,
        camera_transport_delay_ms: float,
        battery_topic: str,
        motor_temperature_topic: str,
        motor_bus_voltage_topic: str,
    ) -> None:
        super().__init__("a1_sim2real_trace_recorder")
        self.csv_path = csv_path
        self.rate_hz = rate_hz
        self.rows = 0
        self.camera_transport_delay_ms = camera_transport_delay_ms
        self.latest: dict[str, Latest] = {
            name: Latest()
            for name in [
                "camera", "camera_raw", "camera_relative", "camera_pose_world",
                "joint", "battery", "motor_temperature", "motor_bus_voltage",
                "fsm", "gate", "policy_enable", "motor_enable", *ARRAY_TOPICS
            ]
        }

        self.create_subscription(PoseStamped, "/pingpong_location", self._camera_cb, qos_profile_sensor_data)
        self.create_subscription(
            PoseStamped,
            "/pingpong_location_raw",
            lambda msg: self._pose_cb("camera_raw", msg),
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/pingpong_location_relative",
            self._camera_relative_cb,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            PoseStamped,
            "/camera_pose_world",
            lambda msg: self._pose_cb("camera_pose_world", msg),
            qos_profile_sensor_data,
        )
        self.create_subscription(JointState, "/right_joint_states", self._joint_cb, qos_profile_sensor_data)
        if battery_topic:
            self.create_subscription(
                BatteryState, battery_topic, self._battery_cb, qos_profile_sensor_data
            )
        if motor_temperature_topic:
            self.create_subscription(
                Float64MultiArray,
                motor_temperature_topic,
                lambda msg: self._array_cb("motor_temperature", msg),
                qos_profile_sensor_data,
            )
        if motor_bus_voltage_topic:
            self.create_subscription(
                Float64MultiArray,
                motor_bus_voltage_topic,
                lambda msg: self._array_cb("motor_bus_voltage", msg),
                qos_profile_sensor_data,
            )
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
        fields += [
            "camera_relative_seq", "camera_relative_recv_epoch_ns",
            "camera_relative_recv_age_ms", "camera_relative_relay_age_ms",
            "camera_relative_transport_delay_ms", "camera_relative_total_age_ms",
            "camera_relative_source_local_ns", "camera_relative_schema",
            "camera_relative_frame", "camera_relative_x", "camera_relative_y",
            "camera_relative_z",
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
        # Append-only telemetry extension: keep every historical column in the
        # same order so existing name- and position-based analysis keeps working.
        for name in ("camera_raw", "camera_pose_world"):
            fields += [
                f"{name}_seq", f"{name}_recv_epoch_ns", f"{name}_recv_age_ms",
                f"{name}_source_ns", f"{name}_source_age_ms", f"{name}_frame",
                f"{name}_x", f"{name}_y", f"{name}_z",
            ]
        fields += [
            "battery_seq", "battery_recv_epoch_ns", "battery_recv_age_ms",
            "battery_source_ns", "battery_source_age_ms", "battery_available",
            "battery_voltage_v", "battery_temperature_c", "battery_current_a",
            "battery_charge_ah", "battery_capacity_ah", "battery_design_capacity_ah",
            "battery_percentage", "battery_power_supply_status",
            "battery_power_supply_health", "battery_power_supply_technology",
            "battery_present", "battery_cell_voltage_json",
            "battery_cell_temperature_json", "battery_location", "battery_serial_number",
        ]
        for name in ("motor_temperature", "motor_bus_voltage"):
            fields += [
                f"{name}_seq", f"{name}_recv_epoch_ns", f"{name}_recv_age_ms",
                f"{name}_available",
            ]
            fields += [f"{name}_{i}" for i in range(1, 8)]
        return fields

    def _camera_cb(self, msg: PoseStamped) -> None:
        self._pose_cb("camera", msg)

    def _pose_cb(self, name: str, msg: PoseStamped) -> None:
        source_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.latest[name].update(
            {
                "source_ns": source_ns,
                "frame": msg.header.frame_id,
                "xyz": [msg.pose.position.x, msg.pose.position.y, msg.pose.position.z],
            }
        )

    def _battery_cb(self, msg: BatteryState) -> None:
        source_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        self.latest["battery"].update(
            {
                "source_ns": source_ns,
                "voltage": msg.voltage,
                "temperature": msg.temperature,
                "current": msg.current,
                "charge": msg.charge,
                "capacity": msg.capacity,
                "design_capacity": msg.design_capacity,
                "percentage": msg.percentage,
                "power_supply_status": msg.power_supply_status,
                "power_supply_health": msg.power_supply_health,
                "power_supply_technology": msg.power_supply_technology,
                "present": int(msg.present),
                "cell_voltage": list(msg.cell_voltage),
                "cell_temperature": list(msg.cell_temperature),
                "location": msg.location,
                "serial_number": msg.serial_number,
            }
        )

    def _camera_relative_cb(self, msg: PoseWithCovarianceStamped) -> None:
        self.latest["camera_relative"].update(
            {
                "relay_age_ms": 1000.0 * msg.pose.covariance[0],
                "schema": msg.pose.covariance[1],
                "frame": msg.header.frame_id,
                "xyz": [
                    msg.pose.pose.position.x,
                    msg.pose.pose.position.y,
                    msg.pose.pose.position.z,
                ],
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

        cam_relative = self.latest["camera_relative"]
        self._slot_base(row, "camera_relative", cam_relative, now_ns)
        if cam_relative.value:
            relay_age_ms = float(cam_relative.value["relay_age_ms"])
            total_age_ms = relay_age_ms + self.camera_transport_delay_ms
            row.update(
                camera_relative_relay_age_ms=relay_age_ms,
                camera_relative_transport_delay_ms=self.camera_transport_delay_ms,
                camera_relative_total_age_ms=total_age_ms,
                camera_relative_source_local_ns=(
                    cam_relative.recv_epoch_ns - int(total_age_ms * 1e6)
                ),
                camera_relative_schema=cam_relative.value["schema"],
                camera_relative_frame=cam_relative.value["frame"],
            )
            for key, value in zip(
                ("camera_relative_x", "camera_relative_y", "camera_relative_z"),
                cam_relative.value["xyz"],
            ):
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

        for name in ("camera_raw", "camera_pose_world"):
            slot = self.latest[name]
            self._slot_base(row, name, slot, now_ns)
            if slot.value:
                source_ns = int(slot.value["source_ns"])
                row.update(
                    {
                        f"{name}_source_ns": source_ns,
                        f"{name}_source_age_ms": (
                            (now_ns - source_ns) / 1e6 if source_ns else ""
                        ),
                        f"{name}_frame": slot.value["frame"],
                    }
                )
                for axis, value in zip(("x", "y", "z"), slot.value["xyz"]):
                    row[f"{name}_{axis}"] = value

        battery = self.latest["battery"]
        self._slot_base(row, "battery", battery, now_ns)
        row["battery_available"] = int(battery.seq > 0)
        if battery.value:
            source_ns = int(battery.value["source_ns"])
            row.update(
                battery_source_ns=source_ns,
                battery_source_age_ms=(now_ns - source_ns) / 1e6 if source_ns else "",
                battery_voltage_v=battery.value["voltage"],
                battery_temperature_c=battery.value["temperature"],
                battery_current_a=battery.value["current"],
                battery_charge_ah=battery.value["charge"],
                battery_capacity_ah=battery.value["capacity"],
                battery_design_capacity_ah=battery.value["design_capacity"],
                battery_percentage=battery.value["percentage"],
                battery_power_supply_status=battery.value["power_supply_status"],
                battery_power_supply_health=battery.value["power_supply_health"],
                battery_power_supply_technology=battery.value["power_supply_technology"],
                battery_present=battery.value["present"],
                battery_cell_voltage_json=json.dumps(battery.value["cell_voltage"]),
                battery_cell_temperature_json=json.dumps(battery.value["cell_temperature"]),
                battery_location=battery.value["location"],
                battery_serial_number=battery.value["serial_number"],
            )

        for name in ("motor_temperature", "motor_bus_voltage"):
            slot = self.latest[name]
            self._slot_base(row, name, slot, now_ns)
            values = slot.value.get("data", [])[:7]
            row[f"{name}_available"] = int(len(values) == 7)
            for i, value in enumerate(values, 1):
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
    parser.add_argument("--camera-transport-delay-ms", type=float, default=15.0)
    parser.add_argument("--battery-topic", default="/battery/state")
    parser.add_argument(
        "--motor-temperature-topic",
        default="",
        help="Optional Float64MultiArray topic ordered r1..r7; blank means unavailable",
    )
    parser.add_argument(
        "--motor-bus-voltage-topic",
        default="",
        help="Optional Float64MultiArray topic ordered r1..r7; blank means unavailable",
    )
    parser.add_argument("--policy-path", default="")
    parser.add_argument("--predictor-path", default="")
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
        "camera_transport_delay_ms": args.camera_transport_delay_ms,
        "csv": str(csv_path),
        "ros_domain_id": os.environ.get("ROS_DOMAIN_ID", ""),
        "rmw_implementation": os.environ.get("RMW_IMPLEMENTATION", ""),
        "cyclonedds_uri": os.environ.get("CYCLONEDDS_URI", ""),
        "topics": {name: topic for name, (topic, _) in ARRAY_TOPICS.items()},
        "passive_topics": {
            "camera_raw": "/pingpong_location_raw",
            "camera_relative": "/pingpong_location_relative",
            "camera_pose_world": "/camera_pose_world",
            "battery": args.battery_topic,
            "motor_temperature": args.motor_temperature_topic,
            "motor_bus_voltage": args.motor_bus_voltage_topic,
        },
        "deployment_artifacts": {
            "policy": _file_identity(args.policy_path),
            "predictor": _file_identity(args.predictor_path),
        },
        "telemetry_contract": {
            "right_joint_states": "q/dq/effort only",
            "battery_state": "pack telemetry only; not per-motor VBUS or temperature",
            "per_motor_temperature": (
                "optional Float64MultiArray r1..r7" if args.motor_temperature_topic else "unavailable"
            ),
            "per_motor_bus_voltage": (
                "optional Float64MultiArray r1..r7" if args.motor_bus_voltage_topic else "unavailable"
            ),
            "landing": (
                "no online inferred label; preserve raw/relative 3D camera tracks for offline "
                "post-contact and landing classification"
            ),
        },
    }
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n")

    rclpy.init(args=ros_args)
    node = Sim2RealTraceRecorder(
        csv_path,
        args.rate_hz,
        args.camera_transport_delay_ms,
        args.battery_topic,
        args.motor_temperature_topic,
        args.motor_bus_voltage_topic,
    )
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    except Exception:
        # CycloneDDS can surface an RCLError from wait_set construction after
        # SIGINT has already invalidated the context. Preserve real runtime
        # errors, but treat that shutdown race as a normal recorder stop.
        if rclpy.ok():
            raise
    finally:
        node.close()
        metadata.update(
            end_epoch_ns=time.time_ns(),
            end_local=datetime.now().astimezone().isoformat(),
            rows=node.rows,
            topic_message_counts={name: slot.seq for name, slot in node.latest.items()},
            telemetry_availability={
                "battery_state": node.latest["battery"].seq > 0,
                "motor_temperature_r1_r7": (
                    len(node.latest["motor_temperature"].value.get("data", [])) >= 7
                ),
                "motor_bus_voltage_r1_r7": (
                    len(node.latest["motor_bus_voltage"].value.get("data", [])) >= 7
                ),
                "camera_raw": node.latest["camera_raw"].seq > 0,
                "camera_relative": node.latest["camera_relative"].seq > 0,
                "camera_pose_world": node.latest["camera_pose_world"].seq > 0,
            },
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
