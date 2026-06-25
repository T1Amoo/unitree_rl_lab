#pragma once
#include <atomic>
#include <array>
#include <chrono>
#include <cmath>
#include <memory>
#include <string>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "tt_ball_source.h"

// Live perception source: subscribes to the mocap topics published by the sim
// (and, on hardware, by a VRPN/Nokov/OptiTrack mocap node). Drop-in replacement
// for ReplayBallSource. heading is left 0 here — the TT state fills env->tt_heading
// from the IMU yaw (the mocap base-quat flips ~180deg on occlusion, so it is unsafe
// as a heading source).
//
// Two site-dependent things are configurable (see State_TableTennis ctor, which
// reads them from config.yaml's TableTennis section):
//   1. Topic names  — sim publishes /mocap/{ball,base}/pose; real VRPN publishes
//      e.g. /vrpn_mocap/U_Tracker0/pose and /vrpn_mocap/g1/pose.
//   2. input_frame transform — the mocap reports in its own room frame M; the
//      policy was trained in world frame W. We map every sample p_W = R_WM*p_M +
//      origin_in_training_world. Defaults are identity/zero, which is exactly the
//      sim case (ros_publish.py already emits in training-world frame), so sim2sim
//      is unaffected when config omits input_frame.
//
// NOTE: deliberately no <Eigen/*> include here — pulling Eigen in after rclcpp in
// this TU triggers a macro clash. The rotation is plain float math instead.
class RosBallSource : public TTBallSource {
public:
    // quat_wxyz: rotation M->W as a (w,x,y,z) quaternion. origin: M-origin in W.
    explicit RosBallSource(rclcpp::Node::SharedPtr node,
                           const std::string& ball_topic = "/mocap/ball/pose",
                           const std::string& base_topic = "/mocap/base/pose",
                           const std::array<float, 4>& quat_wxyz = {1.f, 0.f, 0.f, 0.f},
                           const std::array<float, 3>& origin = {0.f, 0.f, 0.f},
                           int qos_depth = 50)
        : node_(std::move(node)), ox_(origin[0]), oy_(origin[1]), oz_(origin[2]) {
        set_rotation(quat_wxyz);
        // VRPN bursts ~3 PoseStamped frames in <1ms then idles ~10ms; KEEP_LAST(1)
        // would drop 2/3 of every burst at the DDS layer. keep_last(50) buffers the
        // burst. BEST_EFFORT (SensorDataQoS) must match the mocap publisher's QoS.
        auto qos = rclcpp::SensorDataQoS().keep_last(qos_depth);
        ball_sub_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
            ball_topic, qos,
            [this](geometry_msgs::msg::PoseStamped::SharedPtr m) {
                float wx, wy, wz;
                to_training(m->pose.position.x, m->pose.position.y, m->pose.position.z, wx, wy, wz);
                bx_ = wx; by_ = wy; bz_ = wz;
                ball_t_ = now_ns();
            });
        base_sub_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
            base_topic, qos,
            [this](geometry_msgs::msg::PoseStamped::SharedPtr m) {
                float wx, wy, wz;
                to_training(m->pose.position.x, m->pose.position.y, m->pose.position.z, wx, wy, wz);
                rx_ = wx; ry_ = wy; rz_ = wz;
                base_t_ = now_ns();
            });
    }

    TTPerception get(long /*step*/) override {
        TTPerception p;
        p.ball_pos[0] = bx_; p.ball_pos[1] = by_; p.ball_pos[2] = bz_;
        p.robot_pos[0] = rx_; p.robot_pos[1] = ry_; p.robot_pos[2] = rz_;
        p.heading = 0.0f;   // unused: TT state sources heading from the IMU (mocap quat flips)
        long long t = now_ns();
        p.ball_valid = (ball_t_.load() != 0) && (t - ball_t_.load() < STALE_NS);
        p.base_valid = (base_t_.load() != 0) && (t - base_t_.load() < STALE_NS);
        return p;
    }

private:
    // Build the 3x3 rotation matrix from a normalized (w,x,y,z) quaternion.
    void set_rotation(const std::array<float, 4>& q) {
        float w = q[0], x = q[1], y = q[2], z = q[3];
        const float n = std::sqrt(w * w + x * x + y * y + z * z);
        if (n > 1e-9f) { w /= n; x /= n; y /= n; z /= n; }
        else { w = 1.f; x = y = z = 0.f; }
        R_[0][0] = 1 - 2 * (y * y + z * z); R_[0][1] = 2 * (x * y - w * z);     R_[0][2] = 2 * (x * z + w * y);
        R_[1][0] = 2 * (x * y + w * z);     R_[1][1] = 1 - 2 * (x * x + z * z); R_[1][2] = 2 * (y * z - w * x);
        R_[2][0] = 2 * (x * z - w * y);     R_[2][1] = 2 * (y * z + w * x);     R_[2][2] = 1 - 2 * (x * x + y * y);
    }

    // mocap room frame -> training world frame: p_W = R_WM * p_M + origin.
    void to_training(float x, float y, float z, float& ox, float& oy, float& oz) const {
        ox = R_[0][0] * x + R_[0][1] * y + R_[0][2] * z + ox_;
        oy = R_[1][0] * x + R_[1][1] * y + R_[1][2] * z + oy_;
        oz = R_[2][0] * x + R_[2][1] * y + R_[2][2] * z + oz_;
    }

    // 50 ms: mocap runs >=250 Hz so no message for 50 ms means a real dropout.
    static constexpr long long STALE_NS = 50'000'000;

    static long long now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    rclcpp::Node::SharedPtr node_;
    float R_[3][3] = {{1, 0, 0}, {0, 1, 0}, {0, 0, 1}};
    float ox_, oy_, oz_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ball_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr base_sub_;
    std::atomic<float> bx_{0}, by_{0}, bz_{0}, rx_{0}, ry_{0}, rz_{0};
    std::atomic<long long> ball_t_{0}, base_t_{0};  // ns since steady_clock epoch; 0=never received
};
