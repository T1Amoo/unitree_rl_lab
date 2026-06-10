# PerceptionTracker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Insert a robust ball+base perception tracker (gating + Kalman filtering + hysteretic validity state machine + dead-ball/double-bounce/volley logic) between the mocap (`RosBallSource`) and the TT policy, so the deployed robot stops spazzing on ball land/hit/off-table and tolerates real-mocap dropout/jitter/false-points/base-occlusion.

**Architecture:** A header-only, dependency-light C++ class `PerceptionTracker` (Eigen only, no ROS/IsaacLab) ingests per-frame mocap candidates and emits a clean continuous ball position, clean base position, and a debounced `engaged` flag. `State_TableTennis` consumes that instead of raw mocap + a single-frame gate. Because the logic is pure, it is unit-tested standalone (assert-based test executable) before integration; sim2sim fault injection validates end-to-end.

**Tech Stack:** C++17, Eigen3 (`/home/woan/.conda/envs/g1tt_sim2sim/include/eigen3`), CMake (add a `test_perception_tracker` executable like the existing `tt_replay`), Python sim2sim (`ros_publish.py`) for fault injection. Build env: conda `g1tt_sim2sim`.

**Spec:** `docs/superpowers/specs/2026-06-09-perception-tracker-design.md`

**Repo root for paths below:** `/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/g1_23dof`

**Conventions:** world frame, robot faces +X (opponent/serve at +X), robot pelvis at x≈-1.6, table top z=0.76, ball radius 0.02 (bounce at z=0.78), net at x=0, robot own half x∈[-1.37,0]. Control dt=0.02 s (50 Hz). All positions Eigen::Vector3f.

**Build/test command (used in every task):**
```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/g1_23dof
g++ -std=c++17 -O2 -I include -I /home/woan/.conda/envs/g1tt_sim2sim/include/eigen3 \
    test/test_perception_tracker.cpp -o /tmp/tpt && /tmp/tpt
```
Expected on success: prints `ALL TESTS PASSED` and exits 0.

---

## File Structure

- **Create** `include/perception_tracker.h` — the `PerceptionTracker` class + `PTConfig`/`PTOutput` structs. Header-only, Eigen-only. One responsibility: turn raw mocap candidates into a clean, debounced ball/base/engaged signal.
- **Create** `test/test_perception_tracker.cpp` — standalone assert-based unit tests (one `run_*` function per behavior), `main()` calls all and prints `ALL TESTS PASSED`.
- **Modify** `include/FSM/State_TableTennis.h` — replace the raw-ball + raw-robot_pos + single-frame vx/vz `invalid` gate (the block computing `vx,vz,invalid` and the `if(invalid){...}else{...}` around lines ~75-100) with a `PerceptionTracker` member + per-step `update()/output()`.
- **Modify** `CMakeLists.txt` — add a `test_perception_tracker` executable target (optional convenience; the g++ one-liner above is the primary test path).
- **Modify** `sim2sim/ros_publish.py` — add an optional fault-injection mode (env-var gated) to emit dropouts/jitter/false-points for end-to-end validation.

---

## Task 1: PerceptionTracker skeleton + config/output types + playable-volume gate

**Files:**
- Create: `include/perception_tracker.h`
- Create: `test/test_perception_tracker.cpp`

- [ ] **Step 1: Write the failing test**

Create `test/test_perception_tracker.cpp`:
```cpp
#include "perception_tracker.h"
#include <cassert>
#include <cstdio>
#include <cmath>
using Vec3 = Eigen::Vector3f;
static std::vector<Vec3> one(const Vec3& v){ return {v}; }
static const Vec3 BASE(-1.6f, 0.f, 0.793f);

// helper: feed the same incoming ball candidate N frames, return final output
static PTOutput feed(PerceptionTracker& t, const Vec3& ball, int n){
    PTOutput o;
    for(int i=0;i<n;i++){ t.update(one(ball), BASE, true, Vec3::Zero()); o=t.output(); }
    return o;
}

void run_volume_gate(){
    PerceptionTracker t;  // default config
    // A point far outside the playable volume (way behind robot) must never engage.
    PTOutput o = feed(t, Vec3(-5.0f, 0.f, 1.0f), 10);
    assert(o.engaged == false);
    printf("  run_volume_gate OK\n");
}

int main(){
    run_volume_gate();
    printf("ALL TESTS PASSED\n");
    return 0;
}
```

- [ ] **Step 2: Run test to verify it fails**

Run the build/test command above.
Expected: FAIL to compile — `perception_tracker.h` not found / `PerceptionTracker` undefined.

- [ ] **Step 3: Write minimal implementation**

Create `include/perception_tracker.h`:
```cpp
#pragma once
#include <Eigen/Dense>
#include <vector>
#include <deque>
#include <cmath>

struct PTConfig {
    // playable volume (world frame)
    float x_min = -1.4f, x_max = 1.5f, y_abs = 0.95f, z_min = 0.7f, z_max = 2.0f;
    float v_max = 15.0f;          // plausible ball speed (m/s)
    float gate_radius = 0.30f;    // track-continuity gate (m)
    float dt = 0.02f;             // control dt (50 Hz)
    float g = 9.81f;
    // dead-ball
    float roll_z = 0.82f;         // below this + low |vz| = on-table
    float vz_dead = 0.40f;        // |vz| below = not actively bouncing
    float v_rest = 0.30f;         // |v| below = resting
    int   dead_frames = 5;        // persistence for rolling/resting (frames)
    // own half + bounce
    float own_x_lo = -1.37f, own_x_hi = 0.0f;
    float vx_away = 0.30f;        // vx above this = going to opponent
    // hysteresis / coast
    int   confirm_frames = 3;     // K: NO_BALL -> TRACKING
    int   coast_frames = 8;       // M: TRACKING -> NO_BALL
    int   max_coast = 8;          // dropout coast (frames) before track lost
    // ready hold point relative to base (matches deploy: y-offset, height)
    float ready_dy = -0.55f, ready_dz_world = 0.885f;
    int   ramp_frames = 4;
};

struct PTOutput {
    Eigen::Vector3f ball = Eigen::Vector3f::Zero();   // clean ball (world)
    Eigen::Vector3f base = Eigen::Vector3f::Zero();   // clean base (world)
    Eigen::Vector3f prediction_hold = Eigen::Vector3f::Zero(); // ready point (world) when not engaged
    bool engaged = false;
};

class PerceptionTracker {
public:
    explicit PerceptionTracker(PTConfig cfg = PTConfig()) : cfg_(cfg) {}

    // ball_candidates: all mocap ball rigid-body positions this frame (0, 1, or many).
    // base_candidate / has_base: mocap base position this frame.
    // base_vel_odom: leg-odometry base velocity for dropout dead-reckon (Zero() if none).
    void update(const std::vector<Eigen::Vector3f>& ball_candidates,
                const Eigen::Vector3f& base_candidate, bool has_base,
                const Eigen::Vector3f& base_vel_odom);

    PTOutput output() const { return out_; }

private:
    bool in_volume(const Eigen::Vector3f& p) const {
        return p.x() >= cfg_.x_min && p.x() <= cfg_.x_max &&
               std::abs(p.y()) <= cfg_.y_abs &&
               p.z() >= cfg_.z_min && p.z() <= cfg_.z_max &&
               p.allFinite();
    }

    PTConfig cfg_;
    PTOutput out_;
    // (filter + FSM state added in later tasks)
};

// ---- out-of-line definitions ----
inline void PerceptionTracker::update(
        const std::vector<Eigen::Vector3f>& ball_candidates,
        const Eigen::Vector3f& base_candidate, bool has_base,
        const Eigen::Vector3f& base_vel_odom) {
    (void)base_vel_odom;
    if (has_base) out_.base = base_candidate;
    // Minimal: accept a single in-volume candidate as the ball, else not engaged.
    bool any_valid = false;
    for (const auto& c : ball_candidates) {
        if (in_volume(c)) { out_.ball = c; any_valid = true; break; }
    }
    out_.engaged = any_valid;     // refined by gating/FSM in later tasks
    out_.prediction_hold = Eigen::Vector3f(out_.base.x(), out_.base.y() + cfg_.ready_dy, cfg_.ready_dz_world);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run the build/test command. Expected: `run_volume_gate OK` then `ALL TESTS PASSED`, exit 0.

- [ ] **Step 5: Commit**

```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/g1_23dof
git add include/perception_tracker.h test/test_perception_tracker.cpp
git commit -m "feat(tt): PerceptionTracker skeleton + playable-volume gate"
```

---

## Task 2: Constant-velocity-with-gravity Kalman filter (ball smoothing + velocity + coast)

**Files:**
- Modify: `include/perception_tracker.h`
- Test: `test/test_perception_tracker.cpp`

- [ ] **Step 1: Write the failing test** — append to the test file (and add the call in `main()`):
```cpp
void run_kf_smooth_and_coast(){
    PerceptionTracker t;
    // Feed a noisy straight ballistic-ish ball moving toward robot; KF should track it,
    // and during a dropout it should COAST (keep producing a moving estimate, not freeze).
    Vec3 p(1.0f, 0.f, 1.0f); Vec3 v(-3.0f, 0.f, 0.f);
    for(int i=0;i<6;i++){ p += v*0.02f; t.update(one(p), BASE, true, Vec3::Zero()); }
    Vec3 last_seen = t.ball_estimate();
    // dropout: no candidates for 3 frames
    for(int i=0;i<3;i++){ t.update({}, BASE, true, Vec3::Zero()); }
    Vec3 coasted = t.ball_estimate();
    // coasted x must have advanced in -x (moved), not frozen at last_seen
    assert(coasted.x() < last_seen.x() - 0.01f);
    printf("  run_kf_smooth_and_coast OK\n");
}
```
Add `run_kf_smooth_and_coast();` before the print in `main()`.

- [ ] **Step 2: Run test to verify it fails**

Run build/test. Expected: compile error — `ball_estimate()` undefined.

- [ ] **Step 3: Write minimal implementation** — add KF state + methods to `perception_tracker.h`.

Add private members (in the class, after `PTOutput out_;`):
```cpp
    // --- ball KF state (constant velocity + gravity) ---
    bool kf_init_ = false;
    Eigen::Vector3f kf_p_ = Eigen::Vector3f::Zero();   // position estimate
    Eigen::Vector3f kf_v_ = Eigen::Vector3f::Zero();   // velocity estimate
    int miss_ = 0;                                     // consecutive frames with no matched candidate
```
Add public accessor:
```cpp
    Eigen::Vector3f ball_estimate() const { return kf_p_; }
    Eigen::Vector3f ball_velocity() const { return kf_v_; }
```
Add private helpers:
```cpp
    void kf_predict() {            // advance estimate by dt with gravity on z
        kf_p_ += kf_v_ * cfg_.dt;
        kf_p_.z() += 0.5f * (-cfg_.g) * cfg_.dt * cfg_.dt;
        kf_v_.z() += (-cfg_.g) * cfg_.dt;
    }
    void kf_correct(const Eigen::Vector3f& z) {        // simple alpha-beta correction
        const float a = 0.5f, b = 0.3f / cfg_.dt;      // position/velocity blend gains
        Eigen::Vector3f resid = z - kf_p_;
        kf_p_ += a * resid;
        kf_v_ += b * resid * cfg_.dt;
    }
```
Replace the body of `update()` with (keeps the volume gate, adds KF):
```cpp
inline void PerceptionTracker::update(
        const std::vector<Eigen::Vector3f>& ball_candidates,
        const Eigen::Vector3f& base_candidate, bool has_base,
        const Eigen::Vector3f& base_vel_odom) {
    (void)base_vel_odom;
    if (has_base) out_.base = base_candidate;

    // 1) predict
    if (kf_init_) kf_predict();

    // 2) pick a matched candidate (Task 3 adds continuity; here: first in-volume)
    const Eigen::Vector3f* matched = nullptr;
    for (const auto& c : ball_candidates) { if (in_volume(c)) { matched = &c; break; } }

    // 3) correct or coast
    if (matched) {
        if (!kf_init_) { kf_p_ = *matched; kf_v_.setZero(); kf_init_ = true; }
        else kf_correct(*matched);
        miss_ = 0;
    } else {
        miss_++;
        if (miss_ > cfg_.max_coast) kf_init_ = false;   // lost track
    }

    out_.ball = kf_p_;
    out_.engaged = kf_init_;     // refined by FSM in Task 7
    out_.prediction_hold = Eigen::Vector3f(out_.base.x(), out_.base.y() + cfg_.ready_dy, cfg_.ready_dz_world);
}
```

- [ ] **Step 4: Run test to verify it passes**

Run build/test. Expected: `run_kf_smooth_and_coast OK` + `ALL TESTS PASSED`.

- [ ] **Step 5: Commit**
```bash
git add include/perception_tracker.h test/test_perception_tracker.cpp
git commit -m "feat(tt): ball KF (const-vel+gravity) with smoothing and dropout coast"
```

---

## Task 3: Candidate gating — track-continuity + speed + multi-candidate (reflection rejection)

**Files:**
- Modify: `include/perception_tracker.h`
- Test: `test/test_perception_tracker.cpp`

- [ ] **Step 1: Write the failing test** — append + register in `main()`:
```cpp
void run_reflection_rejection(){
    PerceptionTracker t;
    Vec3 p(1.0f,0.f,1.0f), v(-3.0f,0.f,0.f);
    for(int i=0;i<6;i++){ p+=v*0.02f; t.update(one(p), BASE, true, Vec3::Zero()); }
    // Now feed TWO candidates: the true continuation + a far reflection blip.
    p += v*0.02f;
    Vec3 reflection(0.5f, 0.8f, 1.5f);   // in-volume but far from predicted track
    std::vector<Vec3> cands = { reflection, p };
    t.update(cands, BASE, true, Vec3::Zero());
    // tracker must follow the true ball (near p), not jump to the reflection
    assert((t.ball_estimate() - p).norm() < 0.2f);
    assert((t.ball_estimate() - reflection).norm() > 0.5f);
    printf("  run_reflection_rejection OK\n");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run build/test. Expected: assert fails (current code takes the FIRST in-volume candidate = the reflection).

- [ ] **Step 3: Write minimal implementation** — replace the candidate-selection block (step "2) pick a matched candidate") in `update()` with continuity gating:
```cpp
    // 2) match: when a track exists, pick the in-volume candidate closest to the
    //    predicted position within the gate radius (rejects far reflections). When no
    //    track, pick the in-volume candidate with the most plausible speed-to-init.
    const Eigen::Vector3f* matched = nullptr;
    float best = 1e9f;
    for (const auto& c : ball_candidates) {
        if (!in_volume(c)) continue;
        if (kf_init_) {
            float d = (c - kf_p_).norm();
            if (d <= cfg_.gate_radius && d < best) { best = d; matched = &c; }
        } else {
            // no track yet: accept the first in-volume candidate to seed
            matched = &c; break;
        }
    }
```

- [ ] **Step 4: Run test to verify it passes**

Run build/test. Expected: `run_reflection_rejection OK` + `ALL TESTS PASSED`.

- [ ] **Step 5: Commit**
```bash
git add include/perception_tracker.h test/test_perception_tracker.cpp
git commit -m "feat(tt): track-continuity gating rejects reflection candidates"
```

---

## Task 4: Dead-ball detection (out-of-volume, going-away, rolling-on-table, resting)

**Files:**
- Modify: `include/perception_tracker.h`
- Test: `test/test_perception_tracker.cpp`

- [ ] **Step 1: Write the failing test** — append + register:
```cpp
void run_dead_ball(){
    {   // rolling on the table: z at table height, vz~0, moving horizontally -> dead
        PerceptionTracker t;
        Vec3 p(-0.4f, 0.f, 0.785f);             // on table (z≈0.78), in own half
        Vec3 v(-1.0f, 0.f, 0.f);                // rolling toward robot, no vertical motion
        PTOutput o;
        for(int i=0;i<12;i++){ p+=v*0.02f; t.update(one(p),BASE,true,Vec3::Zero()); o=t.output(); }
        assert(o.live == false);                // dead ball -> not a live target
    }
    {   // resting: zero velocity -> dead
        PerceptionTracker t;
        Vec3 p(-0.5f, 0.2f, 0.78f);
        PTOutput o;
        for(int i=0;i<12;i++){ t.update(one(p),BASE,true,Vec3::Zero()); o=t.output(); }
        assert(o.live == false);
    }
    {   // going away (vx>0, post-hit) -> dead
        PerceptionTracker t;
        Vec3 p(-0.5f,0.f,1.0f), v(+4.0f,0.f,0.f);
        PTOutput o;
        for(int i=0;i<12;i++){ p+=v*0.02f; t.update(one(p),BASE,true,Vec3::Zero()); o=t.output(); }
        assert(o.live == false);
    }
    printf("  run_dead_ball OK\n");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run build/test. Expected: compile error — `PTOutput` has no member `live`.

- [ ] **Step 3: Write minimal implementation**

In `PTOutput` add: `bool live = false;` (whether the current ball is a live hittable target, before hysteresis).

Add private state for persistence:
```cpp
    int dead_count_ = 0;    // consecutive frames the ball looks dead (rolling/resting)
```
Add a private helper computing live-ness from the current KF estimate:
```cpp
    bool compute_live() {
        if (!kf_init_) { dead_count_ = 0; return false; }
        const Eigen::Vector3f& p = kf_p_;
        const Eigen::Vector3f& v = kf_v_;
        // out of volume
        if (!in_volume(p)) { dead_count_ = 0; return false; }
        // going away to opponent
        if (v.x() > cfg_.vx_away) return false;
        // rolling on table (low + not bouncing) OR resting (slow), with persistence
        bool on_table = (p.z() < cfg_.roll_z) && (std::abs(v.z()) < cfg_.vz_dead);
        bool resting  = (v.norm() < cfg_.v_rest);
        if (on_table || resting) { if (++dead_count_ >= cfg_.dead_frames) return false; }
        else dead_count_ = 0;
        return true;
    }
```
At the end of `update()` (before assigning `out_.engaged`), set:
```cpp
    out_.live = compute_live();
    out_.engaged = out_.live;    // refined by FSM in Task 7
```

- [ ] **Step 4: Run test to verify it passes**

Run build/test. Expected: `run_dead_ball OK` + `ALL TESTS PASSED`.

- [ ] **Step 5: Commit**
```bash
git add include/perception_tracker.h test/test_perception_tracker.cpp
git commit -m "feat(tt): dead-ball detection (out/away/rolling/resting)"
```

---

## Task 5: Double-bounce detection (one hit attempt per ball)

**Files:**
- Modify: `include/perception_tracker.h`
- Test: `test/test_perception_tracker.cpp`

- [ ] **Step 1: Write the failing test** — append + register:
```cpp
void run_double_bounce(){
    PerceptionTracker t;
    // Simulate two bounces in the own half (x<0) without a paddle hit.
    // Bounce = vz goes negative (down) then positive (up) near table height.
    auto frame=[&](Vec3 p){ t.update(one(p),BASE,true,Vec3::Zero()); };
    // descend to 1st bounce
    frame(Vec3(-0.6f,0.f,1.0f)); frame(Vec3(-0.62f,0.f,0.85f)); frame(Vec3(-0.64f,0.f,0.78f));
    frame(Vec3(-0.66f,0.f,0.85f)); frame(Vec3(-0.68f,0.f,0.95f));   // up (bounce 1)
    frame(Vec3(-0.70f,0.f,0.85f)); frame(Vec3(-0.72f,0.f,0.78f));   // down again
    frame(Vec3(-0.74f,0.f,0.85f));                                  // up (bounce 2)
    // after the 2nd own-half bounce with no paddle hit, ball is dead
    for(int i=0;i<6;i++) frame(Vec3(-0.76f,0.f,0.9f));
    assert(t.output().live == false);
    printf("  run_double_bounce OK\n");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run build/test. Expected: assert fails (no double-bounce logic yet → still live).

- [ ] **Step 3: Write minimal implementation**

Add private state:
```cpp
    int  own_bounce_count_ = 0;
    float prev_vz_ = 0.f;
    bool has_paddle_ = false;     // set by host via set_paddle_hit()
```
Add a public setter the FSM will call when the paddle contacts the ball:
```cpp
public:
    void set_paddle_hit(bool hit) { has_paddle_ = hit; }
private:
```
Add bounce counting at the end of the KF section in `update()` (after `kf_correct`/coast, before `compute_live`):
```cpp
    // bounce detection in own half: vz crosses from descending to ascending near table.
    if (kf_init_) {
        bool ascending_now = kf_v_.z() > 0.3f;
        bool was_descending = prev_vz_ < -0.3f;
        bool near_table = kf_p_.z() < cfg_.roll_z + 0.15f;   // within ~0.15 m of table
        bool own_half = kf_p_.x() < cfg_.own_x_hi;
        if (ascending_now && was_descending && near_table && own_half) own_bounce_count_++;
        // reset when ball goes to opponent half or a new track starts
        if (kf_p_.x() > cfg_.own_x_hi) own_bounce_count_ = 0;
        prev_vz_ = kf_v_.z();
    }
```
Reset `own_bounce_count_ = 0; prev_vz_ = 0.f;` where a new track seeds (in the `if (!kf_init_)` seed branch in `update()`).
In `compute_live()`, add (before the final `return true;`):
```cpp
        if (own_bounce_count_ >= 2 && !has_paddle_) return false;   // double bounce, missed
```

- [ ] **Step 4: Run test to verify it passes**

Run build/test. Expected: `run_double_bounce OK` + `ALL TESTS PASSED`.

- [ ] **Step 5: Commit**
```bash
git add include/perception_tracker.h test/test_perception_tracker.cpp
git commit -m "feat(tt): double-bounce detection (one hit attempt per ball)"
```

---

## Task 6: Volley / out rejection (predict first table contact ∈ own half)

**Files:**
- Modify: `include/perception_tracker.h`
- Test: `test/test_perception_tracker.cpp`

- [ ] **Step 1: Write the failing test** — append + register:
```cpp
void run_volley_reject(){
    {   // flat fast ball that will NOT bounce in own half (flies long/over) -> not live
        PerceptionTracker t;
        Vec3 p(1.2f,0.f,1.05f), v(-7.0f,0.f,0.2f);   // fast, slightly rising -> lands past -1.37
        PTOutput o;
        for(int i=0;i<5;i++){ p+=v*0.02f; v.z()+=(-9.81f)*0.02f; t.update(one(p),BASE,true,Vec3::Zero()); o=t.output(); }
        assert(o.live == false);   // volley/out -> robot must NOT engage
    }
    {   // normal serve that WILL bounce in own half (~x=-0.6) -> live
        PerceptionTracker t;
        Vec3 p(1.0f,0.f,1.0f), v(-4.0f,0.f,1.7f);
        PTOutput o;
        for(int i=0;i<5;i++){ p+=v*0.02f; v.z()+=(-9.81f)*0.02f; t.update(one(p),BASE,true,Vec3::Zero()); o=t.output(); }
        assert(o.live == true);
    }
    printf("  run_volley_reject OK\n");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run build/test. Expected: first assert fails (no volley/out prediction → volley ball is currently live).

- [ ] **Step 3: Write minimal implementation**

Add a private helper that ballistically rolls the current estimate forward to the first
table contact (z=0.78) and returns its x (or +inf if it never descends to the table within
a horizon):
```cpp
    float predicted_first_bounce_x() const {
        // analytic: from kf_p_, kf_v_ under gravity, time to reach z=0.78 (descending root)
        const float zt = 0.78f;
        float a = -0.5f * cfg_.g, b = kf_v_.z(), c = kf_p_.z() - zt;
        float disc = b*b - 4*a*c;
        if (disc < 0) return 1e9f;                  // never reaches table -> treat as out
        float t1 = (-b - std::sqrt(disc)) / (2*a);  // larger positive root (descending)
        float t2 = (-b + std::sqrt(disc)) / (2*a);
        float t = std::max(t1, t2);
        if (t <= 0) return 1e9f;
        return kf_p_.x() + kf_v_.x() * t;
    }
```
In `compute_live()`, gate on the predicted bounce only **before** the ball has bounced in
the own half (once `own_bounce_count_ >= 1`, it's already a legal in-play ball and we keep
engaging through the hit). Add after the going-away check:
```cpp
        if (own_bounce_count_ == 0) {
            float bx = predicted_first_bounce_x();
            // first table contact must be in the own half to be a legal, playable ball
            if (!(bx >= cfg_.own_x_lo && bx <= cfg_.own_x_hi)) return false;  // volley/out
        }
```

- [ ] **Step 4: Run test to verify it passes**

Run build/test. Expected: `run_volley_reject OK` + `ALL TESTS PASSED`.

- [ ] **Step 5: Commit**
```bash
git add include/perception_tracker.h test/test_perception_tracker.cpp
git commit -m "feat(tt): reject volley/out balls (predicted first bounce must be in own half)"
```

---

## Task 7: Hysteretic validity state machine (confirm K / coast M) + engage ramp

**Files:**
- Modify: `include/perception_tracker.h`
- Test: `test/test_perception_tracker.cpp`

- [ ] **Step 1: Write the failing test** — append + register:
```cpp
void run_hysteresis(){
    PerceptionTracker t;   // confirm_frames=3, coast_frames=8 by default
    Vec3 p(1.0f,0.f,1.0f), v(-4.0f,0.f,1.7f);
    auto step=[&](){ p+=v*0.02f; v.z()+=(-9.81f)*0.02f; };
    // 1) a single live frame must NOT engage (needs K=3 consecutive)
    step(); t.update(one(p),BASE,true,Vec3::Zero());
    assert(t.output().engaged == false);
    // 2) after >=3 consecutive live frames, engage
    for(int i=0;i<4;i++){ step(); t.update(one(p),BASE,true,Vec3::Zero()); }
    assert(t.output().engaged == true);
    // 3) a brief 2-frame dropout must NOT disengage (coast M=8)
    t.update({},BASE,true,Vec3::Zero()); t.update({},BASE,true,Vec3::Zero());
    assert(t.output().engaged == true);
    printf("  run_hysteresis OK\n");
}
```

- [ ] **Step 2: Run test to verify it fails**

Run build/test. Expected: assert (1) fails — current `engaged` flips on a single live frame.

- [ ] **Step 3: Write minimal implementation**

Add FSM state:
```cpp
    bool tracking_ = false;       // FSM state: false=NO_BALL(hold), true=TRACKING(engage)
    int  live_run_ = 0;           // consecutive live frames
    int  dead_run_ = 0;           // consecutive not-live frames
```
Replace the final `out_.engaged = out_.live;` with the hysteresis + base-validity gate:
```cpp
    if (out_.live) { live_run_++; dead_run_ = 0; } else { dead_run_++; live_run_ = 0; }
    if (!tracking_ && live_run_ >= cfg_.confirm_frames) tracking_ = true;
    if ( tracking_ && dead_run_ >= cfg_.coast_frames)   tracking_ = false;
    out_.engaged = tracking_ && base_valid_;     // base_valid_ from Task 8 (default true here)
```
Add `bool base_valid_ = true;` to the private members (Task 8 will drive it). Initialize it
`true` so this task's test (which always passes a base) keeps `engaged` controlled by the
ball FSM only.

- [ ] **Step 4: Run test to verify it passes**

Run build/test. Expected: `run_hysteresis OK` + `ALL TESTS PASSED`.

- [ ] **Step 5: Commit**
```bash
git add include/perception_tracker.h test/test_perception_tracker.cpp
git commit -m "feat(tt): hysteretic engage/disengage state machine (confirm K / coast M)"
```

---

## Task 8: BaseTrack — low-pass + jump reject + dropout hold/dead-reckon + lost→disengage

**Files:**
- Modify: `include/perception_tracker.h`
- Test: `test/test_perception_tracker.cpp`

- [ ] **Step 1: Write the failing test** — append + register:
```cpp
void run_base_track(){
    {   // base jitter is smoothed: output base stays near the true slow base
        PerceptionTracker t;
        Vec3 p(1.0f,0.f,1.0f), v(-4.0f,0.f,1.7f);
        for(int i=0;i<6;i++){ p+=v*0.02f; v.z()+=(-9.81f)*0.02f;
            Vec3 jb = BASE + Vec3(0.f, (i%2?0.03f:-0.03f), 0.f);  // ±3cm jitter
            t.update(one(p), jb, true, Vec3::Zero()); }
        assert((t.output().base - BASE).norm() < 0.02f);          // smoothed within 2cm
    }
    {   // base lost too long -> engaged=false even with a perfectly live ball
        PerceptionTracker t;
        Vec3 p(1.0f,0.f,1.0f), v(-4.0f,0.f,1.7f);
        for(int i=0;i<6;i++){ p+=v*0.02f; v.z()+=(-9.81f)*0.02f; t.update(one(p),BASE,true,Vec3::Zero()); }
        assert(t.output().engaged == true);
        for(int i=0;i<12;i++){ p+=v*0.02f; v.z()+=(-9.81f)*0.02f; t.update(one(p),BASE,false,Vec3::Zero()); }
        assert(t.output().engaged == false);                      // base lost -> hold
        printf("  run_base_track OK\n");
    }
}
```

- [ ] **Step 2: Run test to verify it fails**

Run build/test. Expected: first assert may fail (no base smoothing — base copied raw), and/or `base_valid_` never goes false.

- [ ] **Step 3: Write minimal implementation**

Add base state:
```cpp
    bool base_init_ = false;
    Eigen::Vector3f base_p_ = Eigen::Vector3f::Zero();
    int  base_miss_ = 0;
```
Replace the `if (has_base) out_.base = base_candidate;` line at the top of `update()` with a
gated low-pass + dropout coast:
```cpp
    // --- base track: heavy low-pass, jump reject, dropout hold/dead-reckon ---
    if (has_base && base_candidate.allFinite()) {
        if (!base_init_) { base_p_ = base_candidate; base_init_ = true; }
        else {
            float jump = (base_candidate - base_p_).norm();
            if (jump < 0.30f) base_p_ += 0.4f * (base_candidate - base_p_);  // low-pass
            // else: reject as a jump/reflection, keep base_p_ (no update this frame)
        }
        base_miss_ = 0;
    } else {
        base_miss_++;
        base_p_ += base_vel_odom * cfg_.dt;   // dead-reckon (Zero() => hold last)
        if (base_miss_ > cfg_.max_coast) base_init_ = false;
    }
    base_valid_ = base_init_;
    out_.base = base_p_;
```
(Remove the old raw `out_.base = base_candidate;` assignment.) Ensure `out_.engaged =
tracking_ && base_valid_;` (already set in Task 7).

- [ ] **Step 4: Run test to verify it passes**

Run build/test. Expected: `run_base_track OK` + `ALL TESTS PASSED`.

- [ ] **Step 5: Commit**
```bash
git add include/perception_tracker.h test/test_perception_tracker.cpp
git commit -m "feat(tt): base track low-pass + jump reject + dropout coast + lost-disengage"
```

---

## Task 9: Integrate PerceptionTracker into State_TableTennis

**Files:**
- Modify: `include/FSM/State_TableTennis.h`
- Modify: `CMakeLists.txt` (add test target, optional)

- [ ] **Step 1: Read the current ball/gate block**

Run:
```bash
sed -n '60,118p' /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/g1_23dof/include/FSM/State_TableTennis.h
```
Expected: shows the policy-thread loop that reads `ball_src_->get(t)`, computes `vx,vz`,
sets `invalid`, and the `if(invalid){ tt_ball_prediction = ready; predictor_->clear(); }
else { pred = predictor_->update(ball); tt_ball_prediction = pred; }` block.

- [ ] **Step 2: Add the tracker member + include**

At the top includes of `State_TableTennis.h` add:
```cpp
#include "perception_tracker.h"
```
In the private members section add:
```cpp
    PerceptionTracker tracker_;
```

- [ ] **Step 3: Replace the raw gate with the tracker (in the policy-thread loop)**

Replace the block that currently computes `vx, vz, invalid` and the `if(invalid)…else…`
with:
```cpp
                env->robot->update();                              // proprio from DDS
                auto p = ball_src_->get(t);                        // latest mocap
                Eigen::Vector3f ball(p.ball_pos[0], p.ball_pos[1], p.ball_pos[2]);
                Eigen::Vector3f rpos(p.robot_pos[0], p.robot_pos[1], p.robot_pos[2]);
                // feed the robust tracker (single ball candidate from mocap; base from mocap;
                // no leg-odometry yet -> Zero() => hold-last on base dropout).
                bool have_ball = std::isfinite(ball[0]) && (ball.norm() > 1e-6f);
                std::vector<Eigen::Vector3f> cands;
                if (have_ball) cands.push_back(ball);
                tracker_.set_paddle_hit(env->has_touch_paddle);    // if available; else omit
                tracker_.update(cands, rpos, true, Eigen::Vector3f::Zero());
                PTOutput s = tracker_.output();

                env->tt_ball_pos  = s.ball;
                env->tt_robot_pos = s.base;
                auto obs = env->observation_manager->compute();
                auto action = env->alg->act(obs);
                env->action_manager->process_action(action);
                if (s.engaged) {
                    auto pred = predictor_->update({s.ball[0], s.ball[1], s.ball[2]});
                    env->tt_ball_prediction = Eigen::Vector3f(pred[0], pred[1], pred[2]);
                } else {
                    predictor_->clear();
                    env->tt_ball_prediction = s.prediction_hold;
                }
```
Note: if `env->has_touch_paddle` is not exposed in the deploy env, drop the
`set_paddle_hit` line (double-bounce still works via the bounce count; paddle gating is a
refinement). Keep `prev_ball`/`have_prev` only if still referenced elsewhere; otherwise
remove them (they were used by the deleted vx/vz computation).

- [ ] **Step 4: Build the deploy to verify it compiles**

Run:
```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/g1_23dof
bash sim2sim/build_deploy.sh 2>&1 | tail -20
```
Expected: build succeeds, `build/g1_ctrl` rebuilt, no errors referencing PerceptionTracker.
If `has_touch_paddle` is undefined, remove the `set_paddle_hit` line and rebuild.

- [ ] **Step 5: Commit**
```bash
git add include/FSM/State_TableTennis.h
git commit -m "feat(tt): drive policy from PerceptionTracker (clean ball/base + engaged) instead of raw gate"
```

---

## Task 10: sim2sim fault injection + end-to-end validation (GATE: user run)

**Files:**
- Modify: `sim2sim/ros_publish.py`

- [ ] **Step 1: Add env-var-gated fault injection to the mocap publisher**

In `sim2sim/ros_publish.py`, in the `MocapPublisher.publish(...)` path that emits the ball
pose, wrap the published ball position with optional faults read once from env vars:
```python
import os, random
_FAULT = os.environ.get("TT_FAULT", "")          # comma list: drop,jitter,false,baseocc
_t = {"n": 0}
def _apply_ball_faults(pos):
    _t["n"] += 1
    if "jitter" in _FAULT:
        pos = [pos[0]+random.gauss(0,0.01), pos[1]+random.gauss(0,0.01), pos[2]+random.gauss(0,0.01)]
    if "drop" in _FAULT and (_t["n"] // 25) % 4 == 0:   # ~0.5s dropout every 2s
        return None
    return pos
```
Use it where the ball pose is published: if it returns `None`, skip publishing the ball
that frame (simulate dropout). Add a `false` mode that additionally publishes a random
in-volume reflection point, and a `baseocc` mode that skips publishing the pelvis pose for
~0.5 s windows. Keep it OFF when `TT_FAULT` is empty (default).

- [ ] **Step 2: Build + static check**

Run:
```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/Pingpong_TTRL >/dev/null 2>&1 || true
python3 -c "import ast; ast.parse(open('/media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/g1_23dof/sim2sim/ros_publish.py').read()); print('syntax OK')"
```
Expected: `syntax OK`.

- [ ] **Step 3: Commit**
```bash
cd /media/woan/84a38787-1d4e-4ba7-892e-d1d90a009a8c/lgy/unitree_rl_lab/deploy/robots/g1_23dof
git add sim2sim/ros_publish.py
git commit -m "test(tt): sim2sim mocap fault injection (drop/jitter/false/baseocc) via TT_FAULT"
```

- [ ] **Step 4: GATE — user runs sim2sim and confirms each scenario**

User runs (two terminals), pressing `f` then `g` in the MuJoCo window, and confirms the
robot HOLDS (does not spaz/lunge/fall) for each:
```bash
# baseline (no faults): normal rally, robot returns balls
bash sim2sim/run_sim.sh                 # terminal A
bash sim2sim/run_deploy.sh              # terminal B
# then with faults (terminal A):
TT_FAULT=jitter bash sim2sim/run_sim.sh      # smoothed, no spaz
TT_FAULT=drop   bash sim2sim/run_sim.sh      # coast then hold, recovers
TT_FAULT=false  bash sim2sim/run_sim.sh      # reflection ignored
TT_FAULT=baseocc bash sim2sim/run_sim.sh     # robot stays balanced, holds, recovers
```
Plus the serve-driven cases (already in the scene): a ball that flies off the table, a
rolling ball, a double bounce → robot holds in all. Expected: no spaz in any scenario;
normal returns resume when perception is clean.

---

## Self-Review

**1. Spec coverage:**
- §1 problem (spaz from flicker) → Tasks 7 (hysteresis) + 9 (integration). ✓
- §2 safety (mocap loss ≠ fall) → Task 8 (base lost → disengage; balance is proprio, untouched). ✓
- §3 architecture (tracker between source & policy) → Tasks 1-9. ✓
- §4.1 KF → Task 2. §4.2 gating → Tasks 1+3. §4.3 dead-ball → Task 4 (out/away/rolling/resting), Task 5 (double bounce), Task 6 (volley/out). §4.4 lifecycle → emerges from Tasks 4-7. ✓
- §5 BaseTrack → Task 8 (orientation-from-IMU is unchanged in the deploy; tracker only cleans base *position*). ✓
- §6 hysteresis → Task 7. ✓
- §7 integration → Task 9. ✓
- §8 validation → Task 10. ✓
- §9 Tier 3 → explicitly out of scope. ✓

**2. Placeholder scan:** No TBD/"handle edge cases"/uncoded steps — every code step has complete code. ✓

**3. Type consistency:** `PTConfig`, `PTOutput{ball,base,prediction_hold,engaged,live}`, methods `update(cands, base, has_base, base_vel_odom)`, `output()`, `ball_estimate()`, `ball_velocity()`, `set_paddle_hit()`, helpers `in_volume/kf_predict/kf_correct/compute_live/predicted_first_bounce_x`, state `kf_*`, `base_*`, `tracking_/live_run_/dead_run_/base_valid_`, `own_bounce_count_/prev_vz_/has_paddle_`, `dead_count_/miss_/base_miss_` — names consistent across tasks. ✓

**Note for executor:** confirm whether `env->has_touch_paddle` is exposed in the deploy `ManagerBasedRLEnv` (it exists in training). If not, omit `set_paddle_hit` in Task 9 (double-bounce still functions; paddle-gating of the bounce counter is a refinement, not required for v1).
