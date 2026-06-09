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
