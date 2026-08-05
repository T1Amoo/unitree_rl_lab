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

The tracked systemd drop-in also closes a separate camera-control gap.  The
July graph node disables ZED AEC/AGC but does not set exposure or gain, so a
restart can freeze an arbitrary sensor state.  Its `ExecStartPost` now runs
`configure_zed_exposure.sh`, waits until `zed_.open()` has settled, and then
sets and reads back both ZED-X sensors.  The initial moving-ball profile is:

```text
exposure = 6000 us
gain     = 1601 (raw ZED-X V4L2 control, 16.01 dB)
```

The service is fail-closed: a missing/wrong camera or a readback mismatch makes
`ExecStartPost` fail instead of silently running with the old ~15.7 ms
exposure.  A healthy start contains `CAMERA_SETTINGS_OK` in the systemd log.

The age-lock patch is incremental: apply it after `main_graph_low_latency.patch`
or to the currently deployed July source, which already contains that first
overlay. A healthy service prints `[FRESH] locked latest ZED frame` and periodic
`[GRAPH] ... source_age=... grabs=... stale_drop=...` diagnostics. The lock is
fail-closed: an over-age frame never reaches TensorRT or `/pingpong_location`.

Build the patched package in `/home/jetson/pingpong/ros2_ws`; the tracked
systemd drop-in then selects that July workspace instead of the stale June
binary under `/home/jetson/pingpong/install`.

For a live 6/4/2 ms exposure sweep, keep the graph node running and apply each
profile directly to both sensors (the final `0` skips the post-open wait):

```bash
CAM=/home/jetson/unitree_rl_lab_lgy/deploy/robots/a1_h1/jetson_camera/configure_zed_exposure.sh
"$CAM" 6000 1601 0
"$CAM" 4000 1601 0
"$CAM" 2000 1601 0
```

Record about 20 ordinary balls for each setting.  Compare raw
`/pingpong_location` frames per physical trajectory first; do not use policy
reward or robot hits to choose the camera profile.  The selected exposure must
then be written back to the versioned systemd drop-in, committed, and pulled on
Jetson so the next reboot cannot drift.

Apply/build/restart only while the robot policy is disabled. Keep the old
source and installed binary as recoverable artifacts; never delete them.
