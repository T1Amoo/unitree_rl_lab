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
        // DIAG (stage 1): RAW network action, pre scale/offset/clip. The commanded
        // target = scale*raw+offset; for ankle_roll (offset 0, scale 0.25) target =
        // 0.25*raw. Lets us see how far past the clip limit the policy is pushing.
        raw_log_.open("/tmp/tt_raw.csv", std::ios::out | std::ios::trunc);
        if (raw_log_.is_open()) {
            raw_log_ << "t";
            for (int i = 0; i < 23; ++i) raw_log_ << ",raw" << i;
            raw_log_ << "\n";
        }
        // DIAG (stage 1+): NEWEST obs frame (last 87 of the 435 history vector;
        // buffer is oldest..newest so newest = [348:435]). Layout per frame:
        // ang_vel[0:3] proj_grav[3:6] joint_pos_rel[6:29] joint_vel_rel[29:52]
        // last_action[52:75] ball_perception[75:81] ball_pred[81:84] rel_xy[84:86] heading[86].
        // Lets us diff real obs vs the sim ref.npz per-term arrays.
        obs_log_.open("/tmp/tt_obs.csv", std::ios::out | std::ios::trunc);
        if (obs_log_.is_open()) {
            obs_log_ << "t";
            for (int i = 0; i < 87; ++i) obs_log_ << ",o" << i;
            // raw mocap ball (W) + robot_pos (W) + gating flags, to see the actual
            // ball tracking alongside the (gated) obs-frame ball terms.
            obs_log_ << ",bx,by,bz,rx,ry,rz,have_ball,engaged";
            obs_log_ << "\n";
        }
        policy_thread_running = true;
        policy_thread = std::thread([this] {
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desired(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desired);
            auto sleepTill = clock::now() + dt;

            env->reset();
            yaw0_set_ = false;   // capture IMU heading zero-ref on the first frame of this entry
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
                // ball slot [0:3] to the FIXED home sentinel (-2.0,-0.55,0.885) whenever there is
                // no valid live ball (mask_invalid), and feeds the RAW ball only while a ball is
                // live. A PRIOR deploy build instead held a BASE-RELATIVE ready point when idle
                // (out_.prediction_hold, anchored to the moving base) -> self-referential drift ->
                // foot jitter, and was reverted to "raw always" under the mistaken belief that
                // training feeds raw always (it does NOT — it gates to the fixed sentinel). Use
                // the FIXED sentinel here (world/table frame, NOT base-relative) so the idle obs
                // and the serve transition match training exactly. robot slot stays raw (training
                // leaves [3:6] untouched).
                env->tt_ball_pos  = s.engaged ? ball
                                              : Eigen::Vector3f(-1.8f, -0.55f, 0.885f);  // home sentinel when idle (v11: hit_plane_x=-1.8; was -2.0 v7 / -1.6 v5)
                env->tt_robot_pos = rpos;                          // raw base
                // heading = IMU yaw minus a home offset captured at entry. NOT the mocap base quat
                // (p.heading): the mocap rigid-body orientation flips ~180deg on marker occlusion ->
                // garbage heading -> the policy spun the robot around. The IMU yaw is stable (never
                // flips); zeroing it at entry (robot facing the table) makes heading start at 0 ==
                // training heading_w==0, then track real yaw changes drift-free over a short rally.
                {
                    auto& q = env->robot->data.root_quat_w;        // Eigen::Quaternionf (w,x,y,z)
                    float imu_yaw = std::atan2(2.f * (q.w() * q.z() + q.x() * q.y()),
                                               1.f - 2.f * (q.y() * q.y() + q.z() * q.z()));
                    if (!yaw0_set_) { yaw0_ = imu_yaw; yaw0_set_ = true; }
                    float d = imu_yaw - yaw0_;
                    env->tt_heading = std::atan2(std::sin(d), std::cos(d));  // wrap to [-pi, pi]
                }
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
                        constexpr float HOME_X = -1.8f, HOME_Y = 0.0f;
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
                    // (Equivalent to training when the robot is at home x=-2.0; adds restoring.)
                    predictor_->clear();
                    engage_frames_ = 0;
                    constexpr float HOME_X = -1.8f, HOME_Y = 0.0f;   // robot's trained standing base (v11 hit_plane_x=-1.8)
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
                if (raw_log_.is_open()) {
                    raw_log_ << t;
                    for (size_t i = 0; i < action.size(); ++i) raw_log_ << "," << action[i];
                    raw_log_ << "\n";
                    if (t % 50 == 0) raw_log_.flush();
                }
                if (obs_log_.is_open()) {
                    const auto & ov = obs["obs"];
                    if (ov.size() >= 435) {
                        obs_log_ << t;
                        for (int i = 348; i < 435; ++i) obs_log_ << "," << ov[i];
                        obs_log_ << "," << ball[0] << "," << ball[1] << "," << ball[2]
                                 << "," << rpos[0] << "," << rpos[1] << "," << rpos[2]
                                 << "," << (have_ball ? 1 : 0) << "," << (s.engaged ? 1 : 0);
                        obs_log_ << "\n";
                        if (t % 50 == 0) obs_log_.flush();
                    }
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
    std::ofstream raw_log_;    // DIAG (stage 1): per-step RAW network action (pre scale/offset/clip)
    std::ofstream obs_log_;    // DIAG (stage 1+): per-step NEWEST obs frame (87) for sim-vs-real diff
    std::thread policy_thread;
    bool policy_thread_running = false;
    int engage_frames_ = 0;                  // frames since ENGAGE; predictor warm-up gate
    static constexpr int PRED_WARMUP = 5;    // = TTPredictor history_len; hold HOME until filled
    float yaw0_ = 0.f;                        // IMU yaw captured at TableTennis entry (heading zero ref)
    bool yaw0_set_ = false;                   // re-captured each entry so heading starts at 0 facing table
};

REGISTER_FSM(State_TableTennis)
