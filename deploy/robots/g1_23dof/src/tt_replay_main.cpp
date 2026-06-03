// Copyright (c) 2025, Unitree Robotics Co., Ltd.
// All rights reserved.
//
// tt_replay — standalone harness that exercises the REAL deploy obs pipeline
// (ManagerBasedRLEnv + ObservationManager + 4 custom TT obs terms + OrtRunner)
// driven by recorded replay data instead of a live robot. Dumps per-step
// assembled obs (435) and action (23) to .npy for Task-8 equivalence checks.

#include <boost/program_options.hpp>
#include <eigen3/Eigen/Dense>
#include <yaml-cpp/yaml.h>

#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "isaaclab/envs/manager_based_rl_env.h"
#include "isaaclab/algorithms/algorithms.h"
// pull in the registrations for the framework obs/action terms + our TT terms
#include "isaaclab/envs/mdp/observations/observations.h"
#include "isaaclab/envs/mdp/actions/joint_actions.h"
#include "tt_observations.h"
#include "tt_ball_source.h"
#include "tt_predictor.h"

namespace po = boost::program_options;

// ---------------------------------------------------------------------------
// ReplayArticulation: a robot whose `data` is filled from a CSV row instead of
// a live LowState. update() is a no-op — we poke data fields directly each step
// via set_step(). The obs terms read env->robot->data.* exactly as in deploy.
// ---------------------------------------------------------------------------
struct RobotRow {
    float jp[23];
    float jv[23];
    float av[3];
    float pg[3];
    float quat[4];  // qw,qx,qy,qz
    float la[23];
};

class ReplayArticulation : public isaaclab::Articulation {
public:
    void update() override {}  // bypass live-robot read; data set externally

    // Apply one CSV row to data.* . joint_pos is set to jp directly: since the
    // CSV jp_* are already joint_pos_rel (joint_pos - default), the harness sets
    // default_joint_pos to zeros so joint_pos_rel term reproduces jp exactly.
    void set_step(const RobotRow& r) {
        for (int i = 0; i < 23; ++i) {
            data.joint_pos[i] = r.jp[i];
            data.joint_vel[i] = r.jv[i];
        }
        data.root_ang_vel_b = Eigen::Vector3f(r.av[0], r.av[1], r.av[2]);
        data.projected_gravity_b = Eigen::Vector3f(r.pg[0], r.pg[1], r.pg[2]);
        data.root_quat_w = Eigen::Quaternionf(r.quat[0], r.quat[1], r.quat[2], r.quat[3]);
    }
};

// Parse ref_robot.csv (79 cols: jp_0..22, jv_0..22, av0..2, pg0..2, qw..qz, la_0..22)
static std::vector<RobotRow> load_robot_csv(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("cannot open robot csv: " + path);
    std::string line;
    std::getline(f, line);  // header
    std::vector<RobotRow> rows;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<float> vals;
        vals.reserve(79);
        while (std::getline(ss, cell, ',')) vals.push_back(std::stof(cell));
        if (vals.size() != 79)
            throw std::runtime_error("robot csv row has " + std::to_string(vals.size()) +
                                     " cols, expected 79");
        RobotRow r;
        int k = 0;
        for (int i = 0; i < 23; ++i) r.jp[i] = vals[k++];
        for (int i = 0; i < 23; ++i) r.jv[i] = vals[k++];
        for (int i = 0; i < 3; ++i) r.av[i] = vals[k++];
        for (int i = 0; i < 3; ++i) r.pg[i] = vals[k++];
        for (int i = 0; i < 4; ++i) r.quat[i] = vals[k++];
        for (int i = 0; i < 23; ++i) r.la[i] = vals[k++];
        rows.push_back(r);
    }
    return rows;
}

// ---------------------------------------------------------------------------
// Minimal NumPy .npy v1.0 writer for a 2-D row-major float32 array.
// ---------------------------------------------------------------------------
static void write_npy(const std::string& path, const std::vector<float>& data,
                      std::size_t rows, std::size_t cols) {
    std::ostringstream hs;
    hs << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << rows << ", " << cols
       << "), }";
    std::string header = hs.str();
    // total header region (magic 6 + ver 2 + len 2 + header + '\n') padded to 64
    std::size_t prefix = 10;  // magic(6)+major(1)+minor(1)+headerlen(2)
    std::size_t total = prefix + header.size() + 1;  // +1 for trailing '\n'
    std::size_t pad = (64 - (total % 64)) % 64;
    header.append(pad, ' ');
    header.push_back('\n');
    uint16_t hlen = static_cast<uint16_t>(header.size());

    std::ofstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open output: " + path);
    const char magic[] = {'\x93', 'N', 'U', 'M', 'P', 'Y'};
    f.write(magic, 6);
    char ver[2] = {0x01, 0x00};
    f.write(ver, 2);
    f.write(reinterpret_cast<const char*>(&hlen), 2);  // little-endian host
    f.write(header.data(), header.size());
    f.write(reinterpret_cast<const char*>(data.data()), data.size() * sizeof(float));
}

int main(int argc, char** argv) {
    std::string deploy_yaml, policy, predictor, replay, robot, out_obs, out_act;
    long steps = -1;

    po::options_description desc("tt_replay options");
    desc.add_options()("help,h", "help")(
        "deploy_yaml", po::value<std::string>(&deploy_yaml)->required(), "deploy.yaml path")(
        "policy", po::value<std::string>(&policy)->required(), "policy.onnx path")(
        "predictor", po::value<std::string>(&predictor)->required(), "predictor.onnx path")(
        "replay", po::value<std::string>(&replay)->required(), "ref_replay.csv path")(
        "robot", po::value<std::string>(&robot)->required(), "ref_robot.csv path")(
        "out_obs", po::value<std::string>(&out_obs)->required(), "output obs .npy")(
        "out_act", po::value<std::string>(&out_act)->required(), "output action .npy")(
        "steps", po::value<long>(&steps), "number of steps (default: all rows)");

    po::variables_map vm;
    try {
        po::store(po::parse_command_line(argc, argv, desc), vm);
        if (vm.count("help")) { std::cout << desc << "\n"; return 0; }
        po::notify(vm);
    } catch (const std::exception& e) {
        std::cerr << "arg error: " << e.what() << "\n" << desc << "\n";
        return 1;
    }

    // ---- load inputs ----
    auto robot_rows = load_robot_csv(robot);
    ReplayBallSource ball_src(replay);
    long N = static_cast<long>(robot_rows.size());
    if (steps > 0 && steps < N) N = steps;
    std::cout << "[tt_replay] robot rows=" << robot_rows.size()
              << " ball rows=" << ball_src.size() << " using N=" << N << "\n";

    // ---- build env with the replay articulation ----
    auto art = std::make_shared<ReplayArticulation>();
    auto env = std::make_unique<isaaclab::ManagerBasedRLEnv>(YAML::LoadFile(deploy_yaml), art);
    env->alg = std::make_unique<isaaclab::OrtRunner>(policy);

    // jp_* in the CSV are already joint_pos_rel -> zero the default so
    // joint_pos_rel term == jp exactly. (Env ctor set it from yaml default.)
    art->data.default_joint_pos = Eigen::VectorXf::Zero(23);

    TTPredictor predictor_runner(predictor, 5);
    // env->tt_ball_prediction stays zero for t=0 (predictor not yet run).

    // ---- warmup so the history buffers + last_action match the recorder ----
    // The recorder warmed the env before recording; reset() fills each obs
    // term's history with copies of the t=0 single-frame value, and zeroes the
    // action_manager action. Apply t=0 robot+ball state first, then reset.
    art->set_step(robot_rows[0]);
    {
        auto p0 = ball_src.get(0);
        env->tt_ball_pos = Eigen::Vector3f(p0.ball_pos[0], p0.ball_pos[1], p0.ball_pos[2]);
        env->tt_robot_pos = Eigen::Vector3f(p0.robot_pos[0], p0.robot_pos[1], p0.robot_pos[2]);
    }
    // Seed last_action with the recorded la_* of row 0 BEFORE reset() so the
    // history fill (reset re-evaluates each obs term once per history slot) uses
    // it. The recorder's obs[0] last_action slice = the action from the previous
    // (warmup) step, which we mirror with row-0 la. Note: ActionManager::reset()
    // would re-zero _action, so we call process_action AFTER reset (reset only
    // affects the ActionTerm raw buffers, not the obs term histories — those are
    // owned by ObservationManager and were already (re)filled by reset()). To get
    // the seed into the history we instead reset first, then process_action, then
    // re-run obs term history fill via a manual obs-manager reset.
    env->reset();  // global_phase=0, action_manager action=zeros, obs history filled
    env->action_manager->process_action(
        std::vector<float>(robot_rows[0].la, robot_rows[0].la + 23));
    env->observation_manager->reset();  // refill histories with seeded last_action

    std::vector<float> obs_buf;   // N x 435
    std::vector<float> act_buf;   // N x 23
    obs_buf.reserve(static_cast<std::size_t>(N) * 435);
    act_buf.reserve(static_cast<std::size_t>(N) * 23);

    int obs_dim = -1, act_dim = -1;

    for (long t = 0; t < N; ++t) {
        // (a) robot proprio for this step
        art->set_step(robot_rows[t]);
        // (b) ball + robot-base perception for this step
        auto p = ball_src.get(t);
        env->tt_ball_pos = Eigen::Vector3f(p.ball_pos[0], p.ball_pos[1], p.ball_pos[2]);
        env->tt_robot_pos = Eigen::Vector3f(p.robot_pos[0], p.robot_pos[1], p.robot_pos[2]);
        // (c) prediction lag: env->tt_ball_prediction already holds the value
        //     produced at the END of step t-1 (zeros for t=0). Assemble obs now.

        // (d) assemble obs through the REAL ObservationManager (435 floats)
        auto obs = env->observation_manager->compute().at("obs");
        if (obs_dim < 0) {
            obs_dim = static_cast<int>(obs.size());
            std::cout << "[tt_replay] obs dim=" << obs_dim << "\n";
        }
        obs_buf.insert(obs_buf.end(), obs.begin(), obs.end());

        // (f) run policy
        auto action = env->alg->act({{"obs", obs}});
        if (act_dim < 0) {
            act_dim = static_cast<int>(action.size());
            std::cout << "[tt_replay] action dim=" << act_dim << "\n";
        }
        act_buf.insert(act_buf.end(), action.begin(), action.end());

        // (e) make last_action[t+1] == action[t]: push raw net output so the
        //     last_action obs term returns it next step (matches training).
        env->action_manager->process_action(action);

        // (c-next) run predictor AFTER the step (training order) and store for t+1
        auto pred = predictor_runner.update(
            {p.ball_pos[0], p.ball_pos[1], p.ball_pos[2]});
        env->tt_ball_prediction = Eigen::Vector3f(pred[0], pred[1], pred[2]);
    }

    write_npy(out_obs, obs_buf, static_cast<std::size_t>(N),
              static_cast<std::size_t>(obs_dim));
    write_npy(out_act, act_buf, static_cast<std::size_t>(N),
              static_cast<std::size_t>(act_dim));
    std::cout << "[tt_replay] wrote " << out_obs << " (" << N << "x" << obs_dim << ") and "
              << out_act << " (" << N << "x" << act_dim << ")\n";
    return 0;
}
