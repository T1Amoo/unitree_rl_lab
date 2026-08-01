# A1 backhand Jetson camera overlay

This directory keeps the latency/safety overlay for the separately maintained
Jetson `pingpong_detect` package under Git. Do not copy edited source files with
`scp`.

After this repository is pulled on Jetson, validate the patch without changing
the running service:

```bash
cd /home/jetson/pingpong/ros2_ws/src/pingpong_detect
git apply --check \
  /path/to/unitree_rl_lab/deploy/robots/a1_h1/jetson_camera/main_graph_low_latency.patch
git apply --check \
  /path/to/unitree_rl_lab/deploy/robots/a1_h1/jetson_camera/main_graph_exposure_age_lock.patch
```

The overlays do five things:

1. drains already-buffered ZED frames before inference;
2. closes the freshness loop on the ZED IMAGE exposure timestamp: inference is
   allowed only when source age is at most 35 ms; this is above the measured
   23.5--27.4 ms newest-frame floor, while still rejecting the old ~80 ms SDK
   backlog. A callback is dropped after eight unsuccessful grabs instead of
   publishing a stale frame;
3. locks AprilTag calibration after startup;
4. rechecks epipolar geometry after sub-pixel refinement and tightens the
   policy-relevant world corridor;
5. moves per-frame logs to DEBUG so journald does not compete with inference.

The age-lock patch is incremental: apply it after `main_graph_low_latency.patch`
or to the currently deployed July source, which already contains that first
overlay. A healthy service prints `[FRESH] locked latest ZED frame` and periodic
`[GRAPH] ... source_age=... grabs=... stale_drop=...` diagnostics. The lock is
fail-closed: an over-age frame never reaches TensorRT or `/pingpong_location`.

Build the patched package in `/home/jetson/pingpong/ros2_ws`; the tracked
systemd drop-in then selects that July workspace instead of the stale June
binary under `/home/jetson/pingpong/install`.

Apply/build/restart only while the robot policy is disabled. Keep the old
source and installed binary as recoverable artifacts; never delete them.
