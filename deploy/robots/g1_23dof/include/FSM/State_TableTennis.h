#pragma once
#include <thread>
#include <chrono>
#include <memory>
#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include "FSM/FSMState.h"
#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"
#include "isaaclab/algorithms/algorithms.h"
#include "unitree_articulation.h"
#include "tt_observations.h"
#include "tt_ros_ball_source.h"
#include "tt_predictor.h"
#include "perception_tracker.h"

class State_TableTennis : public FSMState
{
public:
    State_TableTennis(int state_mode, std::string state_string)
    : FSMState(state_mode, state_string)
    {
        auto cfg = param::config["FSM"][state_string];
        auto policy_dir = param::parser_policy_dir(cfg["policy_dir"].as<std::string>());

        env = std::make_unique<isaaclab::ManagerBasedRLEnv>(
            YAML::LoadFile(policy_dir / "params" / "deploy.yaml"),
            std::make_shared<unitree::BaseArticulation<LowState_t::SharedPtr>>(FSMState::lowstate));
        env->alg = std::make_unique<isaaclab::OrtRunner>(policy_dir / "exported" / "policy.onnx");
        predictor_ = std::make_unique<TTPredictor>((policy_dir / "exported" / "predictor.onnx").string(), 5);

        // Mocap topic names + room->training-world frame calibration (site config).
        // Defaults = the sim2sim case (/mocap topics, identity transform) so a config
        // without these keys is unchanged; real VRPN overrides them in config.yaml.
        std::string ball_topic = "/mocap/ball/pose";
        std::string base_topic = "/mocap/base/pose";
        if (cfg["ros"]) {
            if (cfg["ros"]["ball_topic"]) ball_topic = cfg["ros"]["ball_topic"].as<std::string>();
            if (cfg["ros"]["base_topic"]) base_topic = cfg["ros"]["base_topic"].as<std::string>();
        }
        std::array<float, 4> quat_wxyz = {1.f, 0.f, 0.f, 0.f};   // rotation M->W (w,x,y,z)
        std::array<float, 3> origin = {0.f, 0.f, 0.f};           // M-origin in W
        if (cfg["input_frame"]) {
            auto fr = cfg["input_frame"];
            if (fr["origin_in_training_world"] && fr["origin_in_training_world"].size() == 3)
                for (int i = 0; i < 3; ++i) origin[i] = fr["origin_in_training_world"][i].as<float>();
            if (fr["rotation_wxyz_to_training"] && fr["rotation_wxyz_to_training"].size() == 4)
                for (int i = 0; i < 4; ++i) quat_wxyz[i] = fr["rotation_wxyz_to_training"][i].as<float>();
        }

        // ROS2 node + spin thread (rclcpp::init() is done in main before the FSM is built)
        ros_node_ = std::make_shared<rclcpp::Node>("g1_tt_deploy");
        ball_src_ = std::make_unique<RosBallSource>(ros_node_, ball_topic, base_topic,
                                                    quat_wxyz, origin);
        std::thread([n = ros_node_]{ rclcpp::spin(n); }).detach();

        registered_checks.emplace_back(std::make_pair(
            [&]() -> bool { return isaaclab::mdp::bad_orientation(env.get(), 1.0); },
            FSMStringMap.right.at("Passive")));
    }

    void enter()
    {
        auto & jmap = env->robot->data.joint_ids_map;
        for (size_t i = 0; i < jmap.size(); ++i) {
            auto & m = lowcmd->msg_.motor_cmd()[jmap[i]];
            m.kp() = env->robot->data.joint_stiffness[i];
            m.kd() = env->robot->data.joint_damping[i];
            m.dq() = 0; m.tau() = 0;
        }
        env->robot->update();

        traj_log_.open("/tmp/arm_mujoco.csv", std::ios::out | std::ios::trunc);
        if (traj_log_.is_open()) {
            traj_log_ << "t";
            for (int i = 0; i < 23; ++i) traj_log_ << ",cmd" << i;
            for (int i = 0; i < 23; ++i) traj_log_ << ",act" << i;
            traj_log_ << "\n";
        }
        policy_thread_running = true;
        policy_thread = std::thread([this] {
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desired(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desired);
            auto sleepTill = clock::now() + dt;

            env->reset();
            long t = 0;
            while (policy_thread_running) {
                env->robot->update();                              // proprio from DDS
                auto p = ball_src_->get(t);                        // latest mocap
                Eigen::Vector3f ball(p.ball_pos[0], p.ball_pos[1], p.ball_pos[2]);
                Eigen::Vector3f rpos(p.robot_pos[0], p.robot_pos[1], p.robot_pos[2]);
                // real freshness from the mocap source (stale topic -> invalid)
                bool have_ball = p.ball_valid && std::isfinite(ball[0]) && (ball.norm() > 1e-6f);
                std::vector<Eigen::Vector3f> cands;
                if (have_ball) cands.push_back(ball);
                tracker_.update(cands, rpos, p.base_valid, Eigen::Vector3f::Zero());
                PTOutput s = tracker_.output();

                // OBS ball-slot gating (match TRAINING): training gates the actor's perception
                // ball slot [0:3] to the FIXED home sentinel (-1.6,-0.55,0.885) whenever there is
                // no valid live ball (mask_invalid), and feeds the RAW ball only while a ball is
                // live. A PRIOR deploy build instead held a BASE-RELATIVE ready point when idle
                // (out_.prediction_hold, anchored to the moving base) -> self-referential drift ->
                // foot jitter, and was reverted to "raw always" under the mistaken belief that
                // training feeds raw always (it does NOT — it gates to the fixed sentinel). Use
                // the FIXED sentinel here (world/table frame, NOT base-relative) so the idle obs
                // and the serve transition match training exactly. robot slot stays raw (training
                // leaves [3:6] untouched).
                env->tt_ball_pos  = s.engaged ? ball
                                              : Eigen::Vector3f(-1.6f, -0.55f, 0.885f);  // home sentinel when idle
                env->tt_robot_pos = rpos;                          // raw base
                auto obs = env->observation_manager->compute();    // uses tt_ball_pos + prev-frame tt_ball_prediction
                auto action = env->alg->act(obs);
                env->action_manager->process_action(action);
                if (s.engaged) {
                    auto pred = predictor_->update({ball[0], ball[1], ball[2]});
                    // Predictor warm-up: ENGAGE clears the history, so TTPredictor left-pads with
                    // the current ball -> a "stationary ball" guess for the first H frames -> a
                    // discontinuous rel_target jump that lurches the policy at serve (the deploy-
                    // side twin of the IsaacLab Bug B). Hold the HOME anchor until H fresh real
                    // frames accumulate, then trust the MLP.
                    if (++engage_frames_ >= PRED_WARMUP) {
                        env->tt_ball_prediction = Eigen::Vector3f(pred[0], pred[1], pred[2]);  // live prediction
                    } else {
                        constexpr float HOME_X = -1.6f, HOME_Y = 0.0f;
                        env->tt_ball_prediction = Eigen::Vector3f(HOME_X, HOME_Y - 0.55f, 0.885f);
                    }
                } else {
                    // No incoming ball: hold the ready target anchored at the robot's HOME
                    // position (fixed in world), NOT the current base. Training masks the
                    // prediction to robot_pos+offset (self-referential) — fine for the very
                    // short gaps training saw, but self-referential has rel_target_x == -0.1
                    // CONSTANT (target always 0.1 m behind) -> no restoring force -> over a
                    // long no-ball gap the robot slowly drifts backward chasing it and falls.
                    // A FIXED home anchor makes rel_target restore toward home -> stable idle.
                    // (Equivalent to training when the robot is at home x=-1.6; adds restoring.)
                    predictor_->clear();
                    engage_frames_ = 0;
                    constexpr float HOME_X = -1.6f, HOME_Y = 0.0f;   // robot's trained standing base
                    env->tt_ball_prediction = Eigen::Vector3f(HOME_X, HOME_Y - 0.55f, 0.885f);
                }

                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
                ++t;
                // DIAG: log commanded (q_des) vs actual joint angles for trajectory
                // comparison with the IsaacLab eval. cmd = processed_actions (scale*a+offset).
                if (traj_log_.is_open()) {
                    auto qdes = env->action_manager->processed_actions();
                    auto & q = env->robot->data.joint_pos;
                    traj_log_ << t;
                    for (size_t i = 0; i < qdes.size(); ++i) traj_log_ << "," << qdes[i];
                    for (int i = 0; i < (int)q.size(); ++i) traj_log_ << "," << q[i];
                    traj_log_ << "\n";
                    if (t % 50 == 0) traj_log_.flush();
                }
            }
        });
    }

    void run()
    {
        auto action = env->action_manager->processed_actions();
        auto & jmap = env->robot->data.joint_ids_map;
        for (size_t i = 0; i < jmap.size(); ++i)
            lowcmd->msg_.motor_cmd()[jmap[i]].q() = action[i];
    }

    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) policy_thread.join();
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;
    std::unique_ptr<TTPredictor> predictor_;
    std::unique_ptr<RosBallSource> ball_src_;
    rclcpp::Node::SharedPtr ros_node_;
    PerceptionTracker tracker_;
    std::ofstream traj_log_;   // DIAG: per-step commanded vs actual joint angles
    std::thread policy_thread;
    bool policy_thread_running = false;
    int engage_frames_ = 0;                  // frames since ENGAGE; predictor warm-up gate
    static constexpr int PRED_WARMUP = 5;    // = TTPredictor history_len; hold HOME until filled
};

REGISTER_FSM(State_TableTennis)
