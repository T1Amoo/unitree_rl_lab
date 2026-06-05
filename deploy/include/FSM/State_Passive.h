// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.

#pragma once

#include "FSMState.h"

class State_Passive : public FSMState
{
public:
    State_Passive(int state, std::string state_string = "Passive") 
    : FSMState(state, state_string) 
    {
        auto motor_mode = param::config["FSM"]["Passive"]["mode"];
        if(motor_mode.IsDefined())
        {
            auto values = motor_mode.as<std::vector<int>>();
            for(int i(0); i<values.size(); ++i)
            {
                lowcmd->msg_.motor_cmd()[i].mode() = values[i];
            }
        }
    } 

    void enter()
    {
        // set gain
        static auto kd = param::config["FSM"]["Passive"]["kd"].as<std::vector<float>>();
        auto kp_node = param::config["FSM"]["Passive"]["kp"];
        static std::vector<float> kp = kp_node.IsDefined()
            ? kp_node.as<std::vector<float>>()
            : std::vector<float>(kd.size(), 0.0f);
        for(int i(0); i < kd.size(); ++i)
        {
            auto & motor = lowcmd->msg_.motor_cmd()[i];
            motor.kp() = (i < (int)kp.size()) ? kp[i] : 0.0f;
            motor.kd() = kd[i];
            motor.dq() = 0;
            motor.tau() = 0;
        }
    }

    void run()
    {
        for(int i(0); i < lowcmd->msg_.motor_cmd().size(); ++i)
        {
            lowcmd->msg_.motor_cmd()[i].q() = lowstate->msg_.motor_state()[i].q();
        }
        // TEMP DEBUG: dump joystick when any tracked button is pressed
        auto & j = lowstate->joystick;
        static int dbg = 0;
        if((j.LT.pressed || j.up.pressed || j.RB.pressed || j.Y.pressed) && (dbg++ % 50 == 0))
            spdlog::info("JOY LT={} up={}(op={}) RB={} Y={}(op={}) timeout={}",
                         j.LT.pressed, j.up.pressed, j.up.on_pressed,
                         j.RB.pressed, j.Y.pressed, j.Y.on_pressed, lowstate->isJoystickTimeout());
    }
};

REGISTER_FSM(State_Passive)
