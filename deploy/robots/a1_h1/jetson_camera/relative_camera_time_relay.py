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

import math

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


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
        if self.min_age_s < 0.0 or self.max_age_s <= self.min_age_s:
            raise ValueError(
                f"invalid relative-age window [{self.min_age_s}, {self.max_age_s}]"
            )

        self.legacy_pub = self.create_publisher(
            PoseStamped, self.legacy_topic, qos_profile_sensor_data
        )
        self.relative_pub = self.create_publisher(
            PoseWithCovarianceStamped, self.relative_topic, qos_profile_sensor_data
        )
        self.subscription = self.create_subscription(
            PoseStamped, self.raw_topic, self._pose_cb, qos_profile_sensor_data
        )
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
