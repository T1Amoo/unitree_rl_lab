# Unified Predictive Validity Gate + No-Ball Idle — Design

**Date:** 2026-06-09
**Status:** Approved (design dialogue complete), ready for plan.
**Scope:** Training env (`Pingpong_TTRL`, non-git) + predictor retrain + deploy alignment (`unitree_rl_lab`).
**Related:** sim2sim/debug.md; docs/superpowers/specs/2026-06-09-perception-tracker-design.md (the deploy PerceptionTracker is the deploy half of the gate).

## 1. Problem

The deployed TT policy DIVERGES (robot drifts backward / "flies off", non-physical) whenever
there is **no incoming ball** (the gap between serves). Confirmed in IsaacLab itself
(`TT_NO_SERVE=1` / `TT_SERVE_PERIOD=10`): with no ball the policy goes unstable. Continuous
serving is rock-stable in both IsaacLab and mujoco. So this is NOT a balance/dynamics gap —
the policy simply has **no learned behavior for the no-ball state**.

## 2. Root cause (verified in code)

- The predictor's regression target is `env.ball_future_pose`
  (on_policy_predictor_regression_runner.py:325/371), which `mask_invalid` replaces with
  `modified_ball_pos` (a ready point) at tt_env.py:1084. So the predictor is ALREADY trained
  to output the ready point when the trajectory is invalid — a "truncation to sentinel" is
  half-built. BUT:
  1. **Training never has a long no-ball period.** The physical ball always exists and is
     re-served within ~1.8 s, so the predictor's INPUT is always a real ball trajectory; it
     never saw "no ball for 5+ s". A long no-ball input is OOD → garbage output → divergence.
  2. **The sentinel is self-referential:** `modified_ball_pos = robot_pos + offset` follows
     the robot, so `rel_target_x == -0.1` CONSTANT (target always 0.1 m behind) → no restoring
     force → over a long gap the robot chases it backward and falls.
  3. **Train/deploy gate mismatch:** training uses `mask_invalid` (reactive: z<0.7, on-floor);
     deploy used a separate ad-hoc gate / manual ready-point override. Two unsynced gates =
     recurring bugs.
- Also (user, real-robot): invalidation must be **predictive** (trajectory will not bounce in
  my half) — waiting for the ball to physically land (z<0.1) is too late and adds a sim2real
  gap.

## 3. Design — one unified predictive validity gate, shared train+deploy

### 3.1 The gate (a deterministic trajectory classifier)
From the (smoothed) ball state (pos + velocity), classify each control step into:
- **VALID-INCOMING:** the ball is a playable incoming ball — predicted (ballistic roll, with a
  margin) to make its first table contact **in the robot's own half** [x∈(-1.37,0)], AND the
  predicted contact/arrival is **within the robot's reach envelope** (reachability gate), AND
  speed is plausible (< v_max), AND not going away (vx below threshold).
- **VALID-AFTER-BOUNCE:** has bounced once in own half and is hittable (not yet double-bounced).
- **INVALID / NO-BALL:** everything else — predicted to fly out/over (volley/out), going away
  (post-hit), out of reach, dead (rolling/resting), double-bounced, or no ball / stale track.

Properties:
- **Predictive, not reactive:** a ball predicted to not bounce in-half is INVALID immediately
  (don't wait for landing). Reactive floor/settle checks are only a backstop.
- **Margins:** the ballistic prediction is sim-physics-based (restitution 0.8, no air
  drag/spin); real balls differ, so all thresholds carry margin and the gate must not flip on
  small prediction error.
- **Hysteresis:** confirm K consecutive VALID frames to engage, coast M frames to disengage —
  identical in train and deploy. Predictive classification needs a few frames to build the
  velocity estimate; hold idle until then.
- **Reachability:** invalidate balls predicted to land beyond the robot's reach — on the real
  robot, never dive for an unreachable ball (safety). Better to concede a point than fall.

### 3.2 Truncation to a FIXED home sentinel (output-side)
When the gate is INVALID/NO-BALL, the predictor output (the policy's `ball_prediction` obs) is
**truncated to a fixed world sentinel = the robot's HOME ready point**:
`SENTINEL = (HOME_X, HOME_Y - 0.55, 0.885)` with HOME = (-1.6, 0) (the trained standing base).
- Fixed (NOT self-referential) → `rel_target` gains a **restoring force** toward home → stable
  idle (no backward runaway).
- Output-side rule (replace prediction with sentinel when invalid) — simplest, deterministic,
  trivially identical in Python and C++.
- The policy learns: prediction = real point → go hit; prediction = home sentinel → idle at home.

### 3.3 Training changes (so the policy LEARNS the no-ball state)
- Replace `mask_invalid` with the unified predictive gate (§3.1); `modified_ball_pos` → the
  fixed home SENTINEL (§3.2).
- **Inject real no-ball periods**: periodically remove the ball (teleport far / parked) for a
  randomized span so the policy + predictor experience long no-ball. Existing posture/balance/
  alive/termination rewards drive a stable idle stance toward the home sentinel (no new reward
  term required initially).
- **Rich invalid coverage:** training serves must include plenty of "flies out / over /
  fast-out / unreachable / no-ball" so the gate's INVALID side and the idle behavior are
  well-learned (not just perfect serves + brief invalid).
- **Retrain the predictor too** (its target now = home sentinel when invalid, and it sees
  no-ball inputs). Warm-start from model_36000.

### 3.4 Deploy alignment
- The deploy gate = the SAME classifier (the PerceptionTracker's logic IS this gate;
  predictive first-bounce + reachability + dead/double-bounce + staleness + hysteresis).
- On INVALID → `tt_ball_prediction = SENTINEL` (same home value as training). Remove ad-hoc
  per-frame overrides; one path.
- `tt_ball_pos`/`tt_robot_pos` = raw mocap always (matches training actor obs).
- The Python (train) and C++ (deploy) gate + ballistic prediction must be **numerically
  equivalent** — single spec, mirrored implementations, a consistency check.

## 4. Decisions locked (from the dialogue)
- Predictive invalidation is the PRIMARY judge (not waiting for landing). ✓
- Sentinel = fixed home (-1.6, -0.55, 0.885). ✓
- Truncation is output-side (rule replaces prediction). ✓
- Reachability gate added (don't dive for unreachable balls). ✓
- Margins + hysteresis for sim2real robustness. ✓
- Rich invalid/no-ball training coverage. ✓
- Predictor retrained alongside the policy (warm-start from 36000). ✓

## 5. Validation
- Env smoke run (few iters) confirms the new gate/sentinel/no-ball injection runs.
- Retrain (warm from 36000) until idle is stable + hitting retained.
- Eval: normal intermittent serving — robot HITS when ball, HOLDS at home when no ball (no
  drift/fall). Use the `TT_NO_SERVE` / `TT_SERVE_PERIOD` hooks to test the no-ball idle.
- sim2sim with the deploy gate aligned: no spaz, no idle drift, returns balls.

## 6. Out of scope (future)
- Real-data predictor fine-tuning / air-drag+spin model (sim2real of the trajectory itself).
- A dedicated idle-stability reward term (only if the existing posture rewards prove
  insufficient).
- Full C++/Python gate auto-sharing (kept as mirrored implementations + consistency test).

## 7. Risks
- The policy may need reward-shaping help to idle (if posture rewards are too weak vs the
  habit of always moving). Mitigation: monitor idle stability during retrain; add a small
  "stay near home + low velocity when invalid" reward if needed.
- Retrain may take several thousand iters to learn idle without losing hitting skill.
- Gate Python/C++ mismatch remains the top integration risk — enforce via shared spec + test.
