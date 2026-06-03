#pragma once
#include <cmath>
#include "isaaclab/envs/mdp/observations/observations.h"   // REGISTER_OBSERVATION + ManagerBasedRLEnv

namespace isaaclab { namespace mdp {

// ball + robot pos in table frame (6) — matches training delayed_perception
REGISTER_OBSERVATION(tt_ball_perception)
{
    return { env->tt_ball_pos[0], env->tt_ball_pos[1], env->tt_ball_pos[2],
             env->tt_robot_pos[0], env->tt_robot_pos[1], env->tt_robot_pos[2] };
}

// learned predictor output (3) — env->tt_ball_prediction is filled by the TT state each step
REGISTER_OBSERVATION(tt_ball_prediction)
{
    return { env->tt_ball_prediction[0], env->tt_ball_prediction[1], env->tt_ball_prediction[2] };
}

// relative target xy (2): [pred_x-0.1, pred_y+0.6] - robot_pos_xy  (training tt_env.py:502-506)
REGISTER_OBSERVATION(tt_rel_target_xy)
{
    float tx = env->tt_ball_prediction[0] - 0.1f;
    float ty = env->tt_ball_prediction[1] + 0.6f;
    return { tx - env->tt_robot_pos[0], ty - env->tt_robot_pos[1] };
}

// heading (1): robot yaw from IMU quaternion (training uses robot.data.heading_w)
REGISTER_OBSERVATION(tt_heading)
{
    auto & q = env->robot->data.root_quat_w;   // Eigen::Quaternionf (w,x,y,z)
    float yaw = std::atan2(2.f * (q.w() * q.z() + q.x() * q.y()),
                           1.f - 2.f * (q.y() * q.y() + q.z() * q.z()));
    return { yaw };
}

}}  // namespace isaaclab::mdp
