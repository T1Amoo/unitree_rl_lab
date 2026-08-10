#!/usr/bin/env python3
"""Capture one robust camera-pose snapshot before the production detector starts.

This helper is run by the project-owned ``a1-camera-pose-snapshot.service``
before ``pingpong-detect.service``.  It starts the existing all-time
calibration executable on private ROS topics, applies the production exposure
profile once the cameras are open, collects a short burst of fresh
``PoseStamped`` samples, writes one atomic cache entry, and then terminates
only the process group it created.

The cached pose is explicitly a startup calibration snapshot.  It is not a
live estimate and must never be presented as one.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import tempfile
import time
from typing import Any, Iterable, Sequence

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data


SCHEMA_VERSION = 1
OUTPUT_FRAME_ID = "world"


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_name, 0o644)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _invalid_payload(reason: str, attempt_wall_time_ns: int) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "valid": False,
        "kind": "startup_camera_pose_snapshot",
        "reason": reason,
        "attempt_wall_time_ns": attempt_wall_time_ns,
        "output_frame_id": OUTPUT_FRAME_ID,
    }


def _median(values: Iterable[float]) -> float:
    return float(statistics.median(values))


def _normalise_quaternion(values: Sequence[float]) -> tuple[float, float, float, float]:
    norm = math.sqrt(sum(value * value for value in values))
    if not math.isfinite(norm) or norm < 1.0e-9:
        raise ValueError("invalid zero/non-finite quaternion")
    return tuple(float(value / norm) for value in values)  # type: ignore[return-value]


def _pose_is_valid(msg: PoseStamped, now_ns: int, max_age_s: float) -> bool:
    stamp_nonzero = bool(msg.header.stamp.sec or msg.header.stamp.nanosec)
    source_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
    age_s = (now_ns - source_ns) / 1.0e9 if source_ns else math.inf
    position = (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z)
    quaternion = (
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
    )
    quaternion_norm = math.sqrt(sum(value * value for value in quaternion))
    return (
        stamp_nonzero
        and msg.header.frame_id == OUTPUT_FRAME_ID
        and math.isfinite(age_s)
        and 0.0 <= age_s <= max_age_s
        and all(math.isfinite(value) for value in position + quaternion)
        and 0.5 <= quaternion_norm <= 1.5
    )


def _aggregate(samples: Sequence[PoseStamped]) -> tuple[dict[str, Any], list[int]]:
    positions = [
        (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z) for msg in samples
    ]
    centre = tuple(_median(position[axis] for position in positions) for axis in range(3))
    residuals = [
        math.sqrt(sum((position[axis] - centre[axis]) ** 2 for axis in range(3)))
        for position in positions
    ]
    residual_median = _median(residuals)
    residual_mad = _median(abs(value - residual_median) for value in residuals)
    # A 3 mm floor avoids rejecting a numerically stable calibration burst merely
    # because its MAD is close to floating-point zero.
    inlier_limit = max(0.003, residual_median + 3.0 * 1.4826 * residual_mad)
    inlier_indices = [
        index for index, residual in enumerate(residuals) if residual <= inlier_limit
    ]
    if not inlier_indices:
        raise ValueError("robust aggregation rejected every sample")

    inlier_positions = [positions[index] for index in inlier_indices]
    robust_position = tuple(
        _median(position[axis] for position in inlier_positions) for axis in range(3)
    )

    quaternions = [
        _normalise_quaternion(
            (
                samples[index].pose.orientation.x,
                samples[index].pose.orientation.y,
                samples[index].pose.orientation.z,
                samples[index].pose.orientation.w,
            )
        )
        for index in inlier_indices
    ]
    reference = quaternions[0]
    aligned_quaternions = []
    for quaternion in quaternions:
        dot = sum(a * b for a, b in zip(reference, quaternion))
        aligned_quaternions.append(
            quaternion if dot >= 0.0 else tuple(-value for value in quaternion)
        )
    robust_quaternion = _normalise_quaternion(
        tuple(
            sum(quaternion[axis] for quaternion in aligned_quaternions)
            for axis in range(4)
        )
    )

    aggregate = {
        "position": {
            "x": robust_position[0],
            "y": robust_position[1],
            "z": robust_position[2],
        },
        "orientation": {
            "x": robust_quaternion[0],
            "y": robust_quaternion[1],
            "z": robust_quaternion[2],
            "w": robust_quaternion[3],
        },
        "position_residual_median_m": residual_median,
        "position_residual_mad_m": residual_mad,
        "position_inlier_limit_m": inlier_limit,
    }
    return aggregate, inlier_indices


class PoseCollector(Node):
    def __init__(self, topic: str, max_age_s: float) -> None:
        super().__init__("a1_camera_pose_startup_probe")
        self.samples: list[PoseStamped] = []
        self.rejected = 0
        self.discard_remaining = 0
        self.max_age_s = max_age_s
        self.subscription = self.create_subscription(
            PoseStamped, topic, self._pose_callback, qos_profile_sensor_data
        )

    def _pose_callback(self, msg: PoseStamped) -> None:
        if not _pose_is_valid(msg, self.get_clock().now().nanoseconds, self.max_age_s):
            self.rejected += 1
            return
        if self.discard_remaining:
            self.discard_remaining -= 1
            return
        self.samples.append(msg)


def _stop_process_group(process: subprocess.Popen[Any], timeout_s: float = 5.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=timeout_s)
        return
    except (ProcessLookupError, subprocess.TimeoutExpired):
        pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=2.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cache-path",
        default="/home/jetson/.cache/a1_camera/camera_pose_world_startup.json",
    )
    parser.add_argument(
        "--config-path",
        default=(
            "/home/jetson/unitree_rl_lab_lgy/deploy/robots/a1_h1/"
            "jetson_camera/detect_config_a1_backhand.yaml"
        ),
    )
    parser.add_argument(
        "--pose-topic", default="/a1_camera_startup_probe/camera_pose_world"
    )
    parser.add_argument(
        "--ball-topic", default="/a1_camera_startup_probe/pingpong_location"
    )
    parser.add_argument("--sample-count", type=int, default=30)
    parser.add_argument("--min-samples", type=int, default=20)
    parser.add_argument("--max-pose-age-s", type=float, default=0.120)
    parser.add_argument("--post-exposure-discard-frames", type=int, default=5)
    parser.add_argument(
        "--exposure-script",
        default=(
            "/home/jetson/unitree_rl_lab_lgy/deploy/robots/a1_h1/"
            "jetson_camera/configure_zed_exposure.sh"
        ),
    )
    parser.add_argument("--exposure-us", type=int, default=6000)
    parser.add_argument("--gain-raw", type=int, default=1601)
    parser.add_argument("--timeout-s", type=float, default=35.0)
    parser.add_argument("--release-wait-s", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    cache_path = Path(args.cache_path)
    attempt_wall_time_ns = time.time_ns()

    # Invalidate the old snapshot before doing anything that may fail.  The
    # production detector is allowed to start after probe failure, but the relay
    # must never republish a previous startup's pose as if it were current.
    try:
        _atomic_write_json(
            cache_path, _invalid_payload("startup probe has not completed", attempt_wall_time_ns)
        )
    except Exception as exc:
        print(f"CAMERA_POSE_STARTUP_INVALIDATE_FAILED: {exc}", flush=True)
        return 0

    if (
        args.sample_count < 1
        or not 1 <= args.min_samples <= args.sample_count
        or args.max_pose_age_s <= 0.0
    ):
        reason = "invalid sample-count/min-samples/max-pose-age configuration"
        _atomic_write_json(cache_path, _invalid_payload(reason, attempt_wall_time_ns))
        print(f"CAMERA_POSE_STARTUP_FAILED: {reason}", flush=True)
        return 0

    command = [
        "ros2",
        "run",
        "pingpong_detect",
        "pingpong_detect_graph_calib_all_time_node",
        "--ros-args",
        "-p",
        f"config_path:={args.config_path}",
        "-r",
        f"/camera_pose_world:={args.pose_topic}",
        "-r",
        f"/pingpong_location:={args.ball_topic}",
    ]

    process: subprocess.Popen[Any] | None = None
    node: PoseCollector | None = None
    rclpy.init(args=[])
    try:
        node = PoseCollector(args.pose_topic, args.max_pose_age_s)
        executor = SingleThreadedExecutor()
        executor.add_node(node)
        process = subprocess.Popen(command, start_new_session=True)
        deadline = time.monotonic() + args.timeout_s

        # The V4L2 controls can only be set after the calibration executable has
        # opened both ZED-X sensors.  Use the first fresh world-frame pose as the
        # readiness signal, apply the same profile as production, then discard
        # the warm-up samples so they cannot enter the robust aggregate.
        while not node.samples and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
            if process.poll() is not None:
                break
        if not node.samples:
            reason = (
                "did not receive a fresh frame_id=world pose before exposure setup "
                f"(rejected {node.rejected}, child_rc {process.poll()})"
            )
            _atomic_write_json(cache_path, _invalid_payload(reason, attempt_wall_time_ns))
            print(f"CAMERA_POSE_STARTUP_FAILED: {reason}", flush=True)
            return 0

        exposure_result = subprocess.run(
            [
                args.exposure_script,
                str(args.exposure_us),
                str(args.gain_raw),
                "0",
            ],
            check=False,
        )
        if exposure_result.returncode != 0:
            reason = f"exposure setup failed with rc={exposure_result.returncode}"
            _atomic_write_json(cache_path, _invalid_payload(reason, attempt_wall_time_ns))
            print(f"CAMERA_POSE_STARTUP_FAILED: {reason}", flush=True)
            return 0
        node.samples.clear()
        node.discard_remaining = max(0, args.post_exposure_discard_frames)

        while len(node.samples) < args.sample_count and time.monotonic() < deadline:
            executor.spin_once(timeout_sec=0.1)
            if process.poll() is not None:
                break

        if len(node.samples) < args.min_samples:
            reason = (
                f"received only {len(node.samples)} valid poses "
                f"(minimum {args.min_samples}, rejected {node.rejected}, "
                f"child_rc {process.poll()})"
            )
            _atomic_write_json(cache_path, _invalid_payload(reason, attempt_wall_time_ns))
            print(f"CAMERA_POSE_STARTUP_FAILED: {reason}", flush=True)
            return 0

        aggregate, inlier_indices = _aggregate(node.samples)
        if len(inlier_indices) < args.min_samples:
            reason = (
                f"only {len(inlier_indices)} robust inliers from {len(node.samples)} poses "
                f"(minimum {args.min_samples})"
            )
            _atomic_write_json(cache_path, _invalid_payload(reason, attempt_wall_time_ns))
            print(f"CAMERA_POSE_STARTUP_FAILED: {reason}", flush=True)
            return 0

        # Keep an actual detector exposure stamp.  The sample nearest the robust
        # position centre is representative and avoids inventing a new timestamp.
        position = aggregate["position"]
        representative_index = min(
            inlier_indices,
            key=lambda index: (
                (node.samples[index].pose.position.x - position["x"]) ** 2
                + (node.samples[index].pose.position.y - position["y"]) ** 2
                + (node.samples[index].pose.position.z - position["z"]) ** 2
            ),
        )
        representative = node.samples[representative_index]
        payload = {
            "schema_version": SCHEMA_VERSION,
            "valid": True,
            "kind": "startup_camera_pose_snapshot",
            "not_realtime": True,
            "attempt_wall_time_ns": attempt_wall_time_ns,
            "completed_wall_time_ns": time.time_ns(),
            "output_frame_id": OUTPUT_FRAME_ID,
            "source_topic": "/camera_pose_world",
            "probe_topic": args.pose_topic,
            "source_header": {
                "frame_id": representative.header.frame_id,
                "stamp": {
                    "sec": int(representative.header.stamp.sec),
                    "nanosec": int(representative.header.stamp.nanosec),
                },
            },
            "pose": {
                "position": aggregate["position"],
                "orientation": aggregate["orientation"],
            },
            "sample_count": len(node.samples),
            "inlier_count": len(inlier_indices),
            "rejected_count": node.rejected,
            "statistics": {
                "position_residual_median_m": aggregate[
                    "position_residual_median_m"
                ],
                "position_residual_mad_m": aggregate["position_residual_mad_m"],
                "position_inlier_limit_m": aggregate["position_inlier_limit_m"],
            },
        }
        _atomic_write_json(cache_path, payload)
        print(
            "CAMERA_POSE_STARTUP_OK "
            f"samples={len(node.samples)} inliers={len(inlier_indices)} "
            f"stamp={representative.header.stamp.sec}."
            f"{representative.header.stamp.nanosec:09d} cache={cache_path}",
            flush=True,
        )
        return 0
    except Exception as exc:
        reason = f"probe exception: {type(exc).__name__}: {exc}"
        try:
            _atomic_write_json(cache_path, _invalid_payload(reason, attempt_wall_time_ns))
        except Exception as cache_exc:
            reason += f"; invalid-cache write also failed: {cache_exc}"
        print(f"CAMERA_POSE_STARTUP_FAILED: {reason}", flush=True)
        return 0
    finally:
        if process is not None:
            try:
                _stop_process_group(process)
            except Exception as exc:
                print(f"CAMERA_POSE_STARTUP_CLEANUP_WARNING: {exc}", flush=True)
        if args.release_wait_s > 0.0:
            time.sleep(args.release_wait_s)
        if node is not None:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    raise SystemExit(main())
