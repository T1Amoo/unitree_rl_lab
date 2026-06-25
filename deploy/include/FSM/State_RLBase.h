// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "isaaclab/envs/mdp/terminations.h"

class State_RLBase : public FSMState
{
public:
    State_RLBase(int state_mode, std::string state_string);
    
    void enter()
    {
        // set gain — map policy-order gains onto SDK motor indices via
        // joint_ids_map (like State_TableTennis::enter). Writing motor_cmd[i]
        // with the raw policy index mis-assigns kp/kd on the sparse 23-DoF SDK
        // enum (arms 15-19/22-26 land on the wrong motors). Correct for 29-DoF
        // too, whose map covers 0..28.
        auto & jmap = env->robot->data.joint_ids_map;
        for (size_t i = 0; i < jmap.size(); ++i)
        {
            auto & m = lowcmd->msg_.motor_cmd()[jmap[i]];
            m.kp() = env->robot->data.joint_stiffness[i];
            m.kd() = env->robot->data.joint_damping[i];
            m.dq() = 0;
            m.tau() = 0;
        }

        env->robot->update();
        // Start policy thread
        policy_thread_running = true;
        policy_thread = std::thread([this]{
            using clock = std::chrono::high_resolution_clock;
            const std::chrono::duration<double> desiredDuration(env->step_dt);
            const auto dt = std::chrono::duration_cast<clock::duration>(desiredDuration);

            // Initialize timing
            auto sleepTill = clock::now() + dt;
            env->reset();

            while (policy_thread_running)
            {
                env->step();

                // Sleep
                std::this_thread::sleep_until(sleepTill);
                sleepTill += dt;
            }
        });
    }

    void run();
    
    void exit()
    {
        policy_thread_running = false;
        if (policy_thread.joinable()) {
            policy_thread.join();
        }
    }

private:
    std::unique_ptr<isaaclab::ManagerBasedRLEnv> env;

    std::thread policy_thread;
    bool policy_thread_running = false;
};

REGISTER_FSM(State_RLBase)
