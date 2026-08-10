#!/usr/bin/env python3
"""Publish project-scoped camera age without comparing two host clocks.

The detector stamps each PoseStamped with the ZED exposure time.  This relay
runs on the same Jetson, so ``now - header.stamp`` is a clock-offset-free frame
age.  It keeps the legacy PoseStamped topic and adds a dedicated relative-time
topic for the A1 ball bridge.

Contract for /pingpong_location_relative (PoseWithCovarianceStamped):
  pose.covariance[0] = exposure-to-relay age in seconds
  pose.covariance[1] = schema version (1.0)
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import (
    DurabilityPolicy,
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
    qos_profile_sensor_data,
)


class RelativeCameraTimeRelay(Node):
    SCHEMA_VERSION = 1.0

    def __init__(self) -> None:
        super().__init__("a1_relative_camera_time_relay")
        self.raw_topic = self.declare_parameter(
            "raw_topic", "/pingpong_location_raw"
        ).value
        self.legacy_topic = self.declare_parameter(
            "legacy_topic", "/pingpong_location"
        ).value
        self.relative_topic = self.declare_parameter(
            "relative_topic", "/pingpong_location_relative"
        ).value
        self.min_age_s = float(self.declare_parameter("min_age_s", 0.005).value)
        self.max_age_s = float(self.declare_parameter("max_age_s", 0.120).value)
        self.diag_every = max(0, int(self.declare_parameter("diag_every", 100).value))
        self.camera_pose_cache_path = str(
            self.declare_parameter(
                "camera_pose_cache_path",
                "/home/jetson/.cache/a1_camera/camera_pose_world_startup.json",
            ).value
        )
        self.camera_pose_topic = str(
            self.declare_parameter("camera_pose_topic", "/camera_pose_world").value
        )
        self.camera_pose_publish_hz = float(
            self.declare_parameter("camera_pose_publish_hz", 1.0).value
        )
        if self.min_age_s < 0.0 or self.max_age_s <= self.min_age_s:
            raise ValueError(
                f"invalid relative-age window [{self.min_age_s}, {self.max_age_s}]"
            )
        if self.camera_pose_publish_hz <= 0.0:
            raise ValueError("camera_pose_publish_hz must be positive")

        self.legacy_pub = self.create_publisher(
            PoseStamped, self.legacy_topic, qos_profile_sensor_data
        )
        self.relative_pub = self.create_publisher(
            PoseWithCovarianceStamped, self.relative_topic, qos_profile_sensor_data
        )
        self.subscription = self.create_subscription(
            PoseStamped, self.raw_topic, self._pose_cb, qos_profile_sensor_data
        )
        self.camera_pose_snapshot = self._load_camera_pose_snapshot(
            Path(self.camera_pose_cache_path)
        )
        self.camera_pose_pub = None
        self.camera_pose_timer = None
        if self.camera_pose_snapshot is not None:
            camera_pose_qos = QoSProfile(
                history=HistoryPolicy.KEEP_LAST,
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            )
            self.camera_pose_pub = self.create_publisher(
                PoseStamped, self.camera_pose_topic, camera_pose_qos
            )
            self.camera_pose_timer = self.create_timer(
                1.0 / self.camera_pose_publish_hz,
                self._publish_camera_pose_snapshot,
            )
            # Publish once immediately; transient-local durability then also
            # serves late subscribers without waiting for the 1 Hz timer.
            self._publish_camera_pose_snapshot()
        self.received = 0
        self.published = 0
        self.dropped = 0
        self.get_logger().info(
            "ready: %s -> legacy=%s relative=%s age=[%.1f, %.1f]ms schema=%.0f"
            % (
                self.raw_topic,
                self.legacy_topic,
                self.relative_topic,
                1000.0 * self.min_age_s,
                1000.0 * self.max_age_s,
                self.SCHEMA_VERSION,
            )
        )

    def _load_camera_pose_snapshot(self, path: Path) -> PoseStamped | None:
        try:
            with path.open("r", encoding="utf-8") as stream:
                payload = json.load(stream)
            if payload.get("schema_version") != 1 or payload.get("valid") is not True:
                reason = payload.get("reason", "cache is not marked valid")
                raise ValueError(str(reason))
            if payload.get("kind") != "startup_camera_pose_snapshot":
                raise ValueError("unexpected cache kind")
            if payload.get("not_realtime") is not True:
                raise ValueError("cache is missing the not_realtime marker")
            if payload.get("output_frame_id") != "world":
                raise ValueError("startup pose output_frame_id must be world")

            stamp = payload["source_header"]["stamp"]
            sec = int(stamp["sec"])
            nanosec = int(stamp["nanosec"])
            if sec < 0 or not 0 <= nanosec < 1_000_000_000 or not (sec or nanosec):
                raise ValueError("invalid/non-exposure source stamp")

            position = payload["pose"]["position"]
            orientation = payload["pose"]["orientation"]
            values = [
                float(position[key]) for key in ("x", "y", "z")
            ] + [float(orientation[key]) for key in ("x", "y", "z", "w")]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("pose contains non-finite values")
            quaternion_norm = math.sqrt(sum(value * value for value in values[3:]))
            if not 0.999 <= quaternion_norm <= 1.001:
                raise ValueError(f"pose quaternion norm is {quaternion_norm:.6f}")

            msg = PoseStamped()
            # Keep the historical /camera_pose_world wire contract.  The stamp
            # is the representative detector exposure timestamp, not relay time.
            msg.header.frame_id = "world"
            msg.header.stamp.sec = sec
            msg.header.stamp.nanosec = nanosec
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z = values[:3]
            (
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ) = values[3:]
            self.get_logger().info(
                "STARTUP_CAMERA_POSE_READY cache=%s exposure_stamp=%d.%09d "
                "samples=%s inliers=%s; snapshot only, NOT realtime"
                % (
                    path,
                    sec,
                    nanosec,
                    payload.get("sample_count", "?"),
                    payload.get("inlier_count", "?"),
                )
            )
            return msg
        except Exception as exc:
            self.get_logger().warning(
                "STARTUP_CAMERA_POSE_UNAVAILABLE cache=%s reason=%s; "
                "/camera_pose_world will not publish"
                % (path, exc)
            )
            return None

    def _publish_camera_pose_snapshot(self) -> None:
        if self.camera_pose_snapshot is not None and self.camera_pose_pub is not None:
            self.camera_pose_pub.publish(self.camera_pose_snapshot)

    def _pose_cb(self, msg: PoseStamped) -> None:
        self.received += 1
        source_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        now_ns = self.get_clock().now().nanoseconds
        age_s = (now_ns - source_ns) / 1e9 if source_ns else math.nan
        if not math.isfinite(age_s) or age_s < self.min_age_s or age_s > self.max_age_s:
            self.dropped += 1
            self.get_logger().warning(
                "drop raw camera sample: Jetson-relative age=%.1fms allowed=[%.1f, %.1f]ms"
                % (1000.0 * age_s, 1000.0 * self.min_age_s, 1000.0 * self.max_age_s),
                throttle_duration_sec=1.0,
            )
            return

        relative = PoseWithCovarianceStamped()
        relative.header = msg.header
        relative.pose.pose = msg.pose
        relative.pose.covariance[0] = age_s
        relative.pose.covariance[1] = self.SCHEMA_VERSION
        self.legacy_pub.publish(msg)
        self.relative_pub.publish(relative)
        self.published += 1

        if self.diag_every and self.published % self.diag_every == 0:
            self.get_logger().info(
                "RELATIVE_CAMERA_TIME_OK age=%.1fms received=%d published=%d dropped=%d"
                % (1000.0 * age_s, self.received, self.published, self.dropped)
            )


def main() -> None:
    rclpy.init()
    node = RelativeCameraTimeRelay()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
