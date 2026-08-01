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
```

The overlay does four things:

1. drains already-buffered ZED frames before inference;
2. locks AprilTag calibration after startup;
3. rechecks epipolar geometry after sub-pixel refinement and tightens the
   policy-relevant world corridor;
4. moves per-frame logs to DEBUG so journald does not compete with inference.

Build the patched package in `/home/jetson/pingpong/ros2_ws`; the tracked
systemd drop-in then selects that July workspace instead of the stale June
binary under `/home/jetson/pingpong/install`.

Apply/build/restart only while the robot policy is disabled. Keep the old
source and installed binary as recoverable artifacts; never delete them.
