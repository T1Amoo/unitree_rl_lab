# PerceptionTracker (G1 TT sim2real ball/base robustness) — Design

**Date:** 2026-06-09
**Status:** Approved (design dialogue complete), ready for implementation plan.
**Author:** design dialogue between user + Claude. See sim2sim/debug.md for project context.

## 1. Problem

The deployed TT policy (v3 = model_36000, rally bounce-trained) "spazzes" (抽风)
when the ball lands / after it's hit / flies off the table. Root cause: the deploy
(`State_TableTennis.h`) feeds the policy **raw mocap** through a **single-frame gate**:

- ball/base positions come straight from `RosBallSource` (mocap rigid bodies);
- ball velocity is a **raw finite difference** of consecutive reads (noisy);
- a single-frame `invalid` predicate flips the predictor between `clear()` (hold) and
  `update()` (track) every frame.

At transitions (hit reverses vx; land crosses z; ball leaves table) the gate **flickers
frame-to-frame**, so the policy's ball/prediction observation **jumps discontinuously**.
The policy was trained on clean continuous trajectories → OOD input → erratic motion.

Real mocap adds more failure modes the current code ignores entirely:
- **Ball:** dropout (occlusion), high-freq jitter, reflective points misdetected as the
  ball, balls that fly off the table / out / roll on the table / double-bounce / volley.
- **Robot base:** the policy obs uses the **absolute mocap `robot_pos`** (3 dims) + a
  relative target term; if the base rigid body is occluded / drifts / jitters / jumps,
  the ball-relative obs is corrupted → spaz, even with perfect ball tracking.

## 2. Safety invariant (must hold)

**Balance/standing runs on the robot's own proprioception (IMU + joint encoders via
LowState), NOT on mocap.** Mocap only feeds the *hitting* obs (absolute ball pos, base
pos, prediction, relative target). Therefore **total mocap loss must NOT make the robot
fall** — it must safely hold a ready stance on proprioception until tracking recovers.

## 3. Architecture

Insert a **`PerceptionTracker`** module between `RosBallSource` and the policy. Raw mocap
is never fed to the policy directly. Each control step (50 Hz) the tracker ingests the
latest mocap candidate(s) and outputs:

- `ball_clean` (3): smoothed/coasted ball position (world frame),
- `base_clean` (3): smoothed/coasted robot base position (world frame),
- `engaged` (bool): whether the policy should actively hit (debounced, hysteretic).

`State_TableTennis` uses these instead of raw ball + raw `robot_pos` + ad-hoc vx/vz gate:
- `engaged == true`  → feed `ball_clean` to the predictor; obs uses `ball_clean`/`base_clean`/prediction.
- `engaged == false` → `predictor.clear()`, obs prediction = **fixed ready point** relative to
  `base_clean` (held constant), transitions **ramped** over a few frames. Robot holds, stays balanced.

Internally the tracker maintains two tracks: **BallTrack** and **BaseTrack**, each with
gating + filtering + a hysteretic validity state machine.

## 4. BallTrack

### 4.1 Filter
Constant-velocity **Kalman filter with gravity** in the predict step (a_z = -g, a_xy = 0).
Smooths jitter, estimates velocity (no raw finite difference), and **predicts forward
during dropouts**. Bounces handled by a process-noise bump / continuing prediction (the
ballistic model is only valid between bounces; the KF re-converges after each bounce).

### 4.2 Candidate gating (per mocap frame; 0, 1, or many candidates)
A candidate is accepted only if ALL:
- **Playable volume:** x ∈ [-1.4, 1.5], |y| ≤ 0.95, z ∈ [0.7, 2.0].
- **Plausible speed:** |v| < V_MAX (≈15 m/s).
- **Track continuity** (if a track exists): within gate radius of the KF-predicted position
  (gate = V_MAX·dt + margin ≈ 0.3 m). Pick the **closest** candidate; reject the rest
  (reflections far from the ball).
- Not NaN/inf/zero-garbage.
New track requires **K consecutive** self-consistent candidates (a lone reflection flicker
won't persist as a coherent trajectory → rejected).

### 4.3 Dead-ball / not-a-live-target detection (→ disengage)
- **Out of volume:** off table / past robot (x<-1.4) / over-high / wide → dead.
- **Going away:** vx > +thresh (post-hit, heading to opponent) sustained → not incoming.
- **Rolling on table:** z < ~0.82 (table top 0.76 + radius) AND |vz| < 0.4, **sustained ≥ N
  frames** → dead. (Persistence distinguishes rolling from a bounce: a bounce has large |vz|
  except for one trough frame.)
- **Resting anywhere (incl. on robot body):** |v| < 0.3 sustained → dead.
- **Double bounce:** count bounces in own half (x<0): a bounce = vz flips −→+ at table
  height. If `own_bounce_count ≥ 2` and `has_touch_paddle == false` → missed → dead.
  Reset count on new serve / ball-to-opponent (x>0) / paddle hit.
- **Volley / out (no legal bounce in own half):** forward-roll the ballistic trajectory to
  the first table contact (z=0.78). If the first contact x ∈ own half [-1.37, 0] → legal
  incoming → OK to engage. If it lands off the end (x<-1.37) / wide / never descends (flat
  volley) → **not a playable ball → hold** (let it go; volleying is illegal + OOD for v3).

### 4.4 Lifecycle (one hit attempt per incoming ball)
```
incoming, predicted 1st bounce ∈ own half  → engage (policy positions)
incoming, predicted bounce out / volley     → hold (let it go)
after 1st own-half bounce                    → engage (hit window)
paddle hit → ball departs (vx>0)             → disengage
2nd own-half bounce (missed)                 → dead → disengage
dropout > coast / out / rolling / resting    → disengage
```

## 5. BaseTrack (robot base = mocap rigid body)

Same gating + filtering, plus proprioception fusion (base is slow & self-sensed):
- **Orientation:** always from the **IMU** (proprioception). Never trust mocap orientation.
- **Position filter:** heavy low-pass / KF with **low process noise** (base moves slowly:
  standing + small steps) → jitter strongly suppressed.
- **Jump rejection:** base cannot teleport → reject candidates outside a small gate around
  the predicted base position (rejects reflections / mislabeled rigid bodies).
- **Dropout:** dead-reckon with **leg odometry** (joint angles + contact feet → base
  velocity) for up to coast time (~150 ms; base moves < a few cm). Optional; "hold last"
  is acceptable v1. Beyond coast → base lost.
- **Base lost too long:** force `engaged=false` → hold; robot stays balanced on
  proprioception (does NOT fall) until base+ball recover.

## 6. Validity state machine (hysteresis — the core anti-flicker fix)

Two states `{NO_BALL(hold) ↔ TRACKING(engage)}`:
- `NO_BALL → TRACKING`: requires the live-ball predicate true for **K consecutive** frames
  (confirm) AND base valid → one reflection blip won't trigger a lunge.
- `TRACKING → NO_BALL`: requires the predicate false for **M consecutive** frames (coast) →
  the brief vx flip at a hit / a few jitter frames / a bounce trough don't drop the track.
- M must exceed a bounce-trough duration (~3-5 frames) but trigger on sustained dead/lost.

`engaged = (ball live & confirmed) AND (base valid)`. On `engaged=false`: predictor.clear(),
prediction = fixed ready point relative to `base_clean`. Engage/disengage **ramped** a few
frames to avoid an obs step.

## 7. Integration points

- New: `include/perception_tracker.h` (header-only C++, Eigen) — `PerceptionTracker` class.
- Modify `include/FSM/State_TableTennis.h`: replace the raw ball + raw robot_pos + vx/vz
  invalid gate (lines ~75-100) with `tracker_.update(candidates, lowstate); auto s =
  tracker_.output();` then set `tt_ball_pos/tt_robot_pos/tt_ball_prediction` from `s` and
  drive the predictor by `s.engaged`.
- `RosBallSource` may need to expose **all** candidate rigid bodies (not just one) so the
  tracker can do reflection rejection. (If mocap publishes one body, single-candidate path.)
- Proprioception (IMU orientation, joint angles, foot contact) already available via
  `env->robot` / LowState.

## 8. Validation (sim2sim fault injection)

Extend `sim2sim/ros_publish.py` (or a wrapper) with a **fault-injection mode** to validate
each defense, then run sim2sim and confirm no spaz:
- serve a ball that **flies off the table** (no in-half bounce) → robot holds, doesn't lunge.
- **rolling ball** on the table → robot holds.
- **double bounce** (disable hitting) → robot holds after 2nd bounce.
- **mocap dropout** (freeze/zero ball or base for N frames) → coast then hold, recover cleanly.
- **jitter** (gaussian noise on ball/base) → smoothed, no spaz.
- **false points** (inject spurious reflective points) → rejected, robot ignores.
- **base occlusion** (drop robot_pos for a span) → robot stays balanced, holds, recovers.

## 9. Tier 3 (future, training-side; out of scope for this implementation)

Domain-randomize perception in training (g1_tt): random ball/base **dropout**, position
**jitter**, occasional **false blips**, wider **latency**. Fine-tune v3 (warm-start) so the
policy itself tolerates imperfect perception (belt-and-suspenders with the deploy tracker).
Tracked as a follow-up; this spec covers the **deploy-side Tier 1+2** tracker only.

## 10. Out of scope
- Tier 3 retraining (separate spec/plan).
- Multi-ball game logic beyond "track the one valid incoming ball".
- Leg-odometry base estimator is optional for v1 ("hold last" acceptable); can be a follow-up.
