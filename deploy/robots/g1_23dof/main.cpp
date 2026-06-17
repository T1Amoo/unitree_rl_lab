#include "FSM/CtrlFSM.h"
#include "FSM/State_Passive.h"
#include "FSM/State_FixStand.h"
#include "FSM/State_RLBase.h"
#include "FSM/State_TableTennis.h"
#include <rclcpp/rclcpp.hpp>

std::unique_ptr<LowCmd_t> FSMState::lowcmd = nullptr;
std::shared_ptr<LowState_t> FSMState::lowstate = nullptr;
std::shared_ptr<Keyboard> FSMState::keyboard = nullptr;

void init_fsm_state()
{
    auto lowcmd_sub = std::make_shared<unitree::robot::g1::subscription::LowCmd>();
    usleep(0.2 * 1e6);
    if(!lowcmd_sub->isTimeout())
    {
        spdlog::critical("The other process is using the lowcmd channel, please close it first.");
        unitree::robot::go2::shutdown();
        // exit(0);
    }
    FSMState::lowcmd = std::make_unique<LowCmd_t>();
    FSMState::lowstate = std::make_shared<LowState_t>();
    spdlog::info("Waiting for connection to robot...");
    FSMState::lowstate->wait_for_connection();
    spdlog::info("Connected to robot.");
}

int main(int argc, char** argv)
{
    // Load parameters
    auto vm = param::helper(argc, argv);

    std::cout << " --- Unitree Robotics --- \n";
    std::cout << "     G1-23dof Controller \n";

    // Unitree DDS Config. unitree_mujoco sim2sim runs on domain 1 over loopback
    // (--network lo); the real G1 runs on domain 0 over its NIC (--network eth0 /
    // enpXs0). Pick the domain from the interface so one binary serves both without
    // a recompile.
    const std::string network = vm["network"].as<std::string>();
    const bool local_sim = (network == "lo" || network == "lo0");
    unitree::robot::ChannelFactory::Instance()->Init(local_sim ? 1 : 0, network);

    init_fsm_state();

    FSMState::lowcmd->msg_.mode_machine() = 4; // 23dof
    if(!FSMState::lowcmd->check_mode_machine(FSMState::lowstate)) {
        spdlog::critical("Unmatched robot type.");
        exit(-1);
    }

    rclcpp::init(argc, argv);

    // Initialize FSM
    auto fsm = std::make_unique<CtrlFSM>(param::config["FSM"]);
    fsm->start();

    std::cout << "Press [L2 + Up] to enter FixStand mode.\n";
    std::cout << "And then press [R1 + X] to start controlling the robot.\n";
    std::cout << "Press [R1 + Y] to enter TableTennis mode.\n";

    while (true)
    {
        sleep(1);
    }
    
    return 0;
}

