#pragma once
#include <thread>
#include <chrono>
#include <memory>
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

        // ROS2 node + spin thread (rclcpp::init() is done in main before the FSM is built)
        ros_node_ = std::make_shared<rclcpp::Node>("g1_tt_deploy");
        ball_src_ = std::make_unique<RosBallSource>(ros_node_);
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

        policy_thread_running = true;
        policy_thread = std::thread([this] {
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desired(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desired);
            auto sleepTill = clock::now() + dt;

            env->reset();
            long t = 0;
            Eigen::Vector3f prev_ball = Eigen::Vector3f::Zero();
            bool have_prev = false;
            while (policy_thread_running) {
                env->robot->update();                              // proprio from DDS
                auto p = ball_src_->get(t);                        // latest mocap
                Eigen::Vector3f ball(p.ball_pos[0], p.ball_pos[1], p.ball_pos[2]);
                Eigen::Vector3f rpos(p.robot_pos[0], p.robot_pos[1], p.robot_pos[2]);
                env->tt_ball_pos  = ball;
                env->tt_robot_pos = rpos;
                // ball velocity from consecutive deploy reads (for the invalid gate)
                float vx = 0.f, vz = 0.f;
                if (have_prev) { vx = (ball[0]-prev_ball[0])/env->step_dt; vz = (ball[2]-prev_ball[2])/env->step_dt; }
                prev_ball = ball; have_prev = true;
                // mask_invalid (matches training tt_env.py:1044): ball is NOT a live
                // incoming serve -> do not chase. Real-robot-safe: otherwise the robot
                // lunges at a landed / returning / out-of-range ball and topples.
                bool invalid = (ball[0] < -1.9f) || (vx > 0.3f) || (ball[2] < 0.7f)
                               || (ball[0] < -1.35f && vz < 0.f);
                auto obs = env->observation_manager->compute();    // uses prev-step prediction
                auto action = env->alg->act(obs);
                env->action_manager->process_action(action);       // also feeds last_action
                if (invalid) {
                    // hold point relative to robot (training modified_ball_pos:
                    // robot_y + paddle_y_offset(-0.55), z = body_height(0.685)+0.2)
                    // -> rel_target ~ ready stance, robot holds instead of chasing.
                    env->tt_ball_prediction = Eigen::Vector3f(rpos[0], rpos[1] - 0.55f, 0.885f);
                    predictor_->clear();
                } else {
                    auto pred = predictor_->update({ball[0], ball[1], ball[2]});
                    env->tt_ball_prediction = Eigen::Vector3f(pred[0], pred[1], pred[2]);
                }

                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
                ++t;
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
    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_TableTennis)
