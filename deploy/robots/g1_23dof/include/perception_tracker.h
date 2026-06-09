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
    bool live = false;
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
    Eigen::Vector3f ball_estimate() const { return kf_p_; }
    Eigen::Vector3f ball_velocity() const { return kf_v_; }
    void set_paddle_hit(bool hit) { has_paddle_ = hit; }

private:
    bool in_volume(const Eigen::Vector3f& p) const {
        return p.x() >= cfg_.x_min && p.x() <= cfg_.x_max &&
               std::abs(p.y()) <= cfg_.y_abs &&
               p.z() >= cfg_.z_min && p.z() <= cfg_.z_max &&
               p.allFinite();
    }
    void kf_predict() {            // advance estimate by dt with gravity on z
        kf_p_ += kf_v_ * cfg_.dt;
        kf_p_.z() += 0.5f * (-cfg_.g) * cfg_.dt * cfg_.dt;
        kf_v_.z() += (-cfg_.g) * cfg_.dt;
    }
    void kf_correct(const Eigen::Vector3f& z) {        // simple alpha-beta correction
        const float a = 0.5f, b = 0.3f / cfg_.dt;      // position/velocity blend gains
        Eigen::Vector3f resid = z - kf_p_;
        kf_p_ += a * resid;
        kf_v_ += b * resid;
    }

    PTConfig cfg_;
    PTOutput out_;
    // --- ball KF state (constant velocity + gravity) ---
    bool kf_init_ = false;
    Eigen::Vector3f kf_p_ = Eigen::Vector3f::Zero();   // position estimate
    Eigen::Vector3f kf_v_ = Eigen::Vector3f::Zero();   // velocity estimate
    int miss_ = 0;                                     // consecutive frames with no matched candidate
    int dead_count_ = 0;   // consecutive frames the ball looks dead (rolling/resting)
    int  own_bounce_count_ = 0;
    float prev_vz_ = 0.f;
    bool has_paddle_ = false;     // set by host via set_paddle_hit()

    bool compute_live() {
        if (!kf_init_) { dead_count_ = 0; return false; }
        const Eigen::Vector3f& p = kf_p_;
        const Eigen::Vector3f& v = kf_v_;
        if (!in_volume(p)) { dead_count_ = 0; return false; }        // out of volume
        if (v.x() > cfg_.vx_away) return false;                       // going away to opponent
        bool on_table = (p.z() < cfg_.roll_z) && (std::abs(v.z()) < cfg_.vz_dead);
        bool resting  = (v.norm() < cfg_.v_rest);
        if (on_table || resting) { if (++dead_count_ >= cfg_.dead_frames) return false; }
        else dead_count_ = 0;
        if (own_bounce_count_ >= 2 && !has_paddle_) return false;   // double bounce, missed
        return true;
    }
};

// ---- out-of-line definitions ----
inline void PerceptionTracker::update(
        const std::vector<Eigen::Vector3f>& ball_candidates,
        const Eigen::Vector3f& base_candidate, bool has_base,
        const Eigen::Vector3f& base_vel_odom) {
    (void)base_vel_odom;
    if (has_base) out_.base = base_candidate;

    // 1) predict
    if (kf_init_) kf_predict();

    // 2) match: when a track exists, pick the in-volume candidate closest to the
    //    predicted position within the gate radius (rejects far reflections). When no
    //    track, pick the first in-volume candidate to seed.
    const Eigen::Vector3f* matched = nullptr;
    float best = 1e9f;
    for (const auto& c : ball_candidates) {
        if (!in_volume(c)) continue;
        if (kf_init_) {
            float d = (c - kf_p_).norm();
            if (d <= cfg_.gate_radius && d < best) { best = d; matched = &c; }
        } else {
            matched = &c; break;   // no track yet: seed with first in-volume candidate
        }
    }

    // 3) correct or coast
    if (matched) {
        if (!kf_init_) { kf_p_ = *matched; kf_v_.setZero(); kf_init_ = true; own_bounce_count_ = 0; prev_vz_ = 0.f; }
        else kf_correct(*matched);
        miss_ = 0;
    } else {
        miss_++;
        if (miss_ > cfg_.max_coast) kf_init_ = false;   // lost track
    }

    // bounce detection in own half: vz crosses from descending to ascending near table.
    if (kf_init_) {
        bool ascending_now = kf_v_.z() > 0.05f;   // zero-crossing: no longer descending (was: 0.3, too strict vs KF lag)
        bool was_descending = prev_vz_ < -0.3f;
        bool near_table = kf_p_.z() < cfg_.roll_z + 0.15f;   // within ~0.15 m of table
        bool own_half = kf_p_.x() < cfg_.own_x_hi;
        if (ascending_now && was_descending && near_table && own_half) own_bounce_count_++;
        if (kf_p_.x() > cfg_.own_x_hi) own_bounce_count_ = 0;   // reset in opponent half
        prev_vz_ = kf_v_.z();
    }

    out_.ball = kf_p_;
    out_.live = compute_live();
    out_.engaged = out_.live;    // refined by FSM in Task 7
    out_.prediction_hold = Eigen::Vector3f(out_.base.x(), out_.base.y() + cfg_.ready_dy, cfg_.ready_dz_world);
}
