#pragma once
#include <atomic>
#include <chrono>
#include <memory>
#include <rclcpp/rclcpp.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include "tt_ball_source.h"

// Live perception source: subscribes to the mocap topics published by the sim
// (and, on hardware, by Nokov). Drop-in replacement for ReplayBallSource.
// heading is left 0 — deploy sources heading from IMU via the tt_heading obs term.
class RosBallSource : public TTBallSource {
public:
    explicit RosBallSource(rclcpp::Node::SharedPtr node) : node_(std::move(node)) {
        ball_sub_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/mocap/ball/pose", rclcpp::SensorDataQoS(),
            [this](geometry_msgs::msg::PoseStamped::SharedPtr m) {
                bx_ = m->pose.position.x; by_ = m->pose.position.y; bz_ = m->pose.position.z;
                ball_t_ = now_ns();
            });
        base_sub_ = node_->create_subscription<geometry_msgs::msg::PoseStamped>(
            "/mocap/base/pose", rclcpp::SensorDataQoS(),
            [this](geometry_msgs::msg::PoseStamped::SharedPtr m) {
                rx_ = m->pose.position.x; ry_ = m->pose.position.y; rz_ = m->pose.position.z;
                base_t_ = now_ns();
            });
    }

    TTPerception get(long /*step*/) override {
        TTPerception p;
        p.ball_pos[0] = bx_; p.ball_pos[1] = by_; p.ball_pos[2] = bz_;
        p.robot_pos[0] = rx_; p.robot_pos[1] = ry_; p.robot_pos[2] = rz_;
        p.heading = 0.0f;
        long long t = now_ns();
        p.ball_valid = (ball_t_.load() != 0) && (t - ball_t_.load() < STALE_NS);
        p.base_valid = (base_t_.load() != 0) && (t - base_t_.load() < STALE_NS);
        return p;
    }

private:
    // 50 ms: mocap runs >=250 Hz so no message for 50 ms means a real dropout.
    static constexpr long long STALE_NS = 50'000'000;

    static long long now_ns() {
        return std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    rclcpp::Node::SharedPtr node_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr ball_sub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr base_sub_;
    std::atomic<float> bx_{0}, by_{0}, bz_{0}, rx_{0}, ry_{0}, rz_{0};
    std::atomic<long long> ball_t_{0}, base_t_{0};  // ns since steady_clock epoch; 0=never received
};
