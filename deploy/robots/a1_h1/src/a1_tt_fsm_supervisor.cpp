#include <algorithm>
#include <array>
#include <cmath>
#include <cctype>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/int32.hpp>
#include <std_msgs/msg/string.hpp>

namespace {

constexpr size_t kNumJoints = 7;
constexpr double kPi = 3.14159265358979323846;

const std::array<double, kNumJoints> kDefaultRightQ = {
    1.450, -0.762, -2.050, 1.445, 0.206, -0.827, 1.043};
const std::array<double, kNumJoints> kRightQMin = {
    -1.05, -3.14, -2.76, -1.92, -2.76, -1.57, -2.76};
const std::array<double, kNumJoints> kRightQMax = {
    3.14, 0.262, 2.76, 1.92, 2.76, 1.57, 2.76};
const std::array<std::array<const char*, 3>, kNumJoints> kRightJointAliases = {{
    {{"joint1-a1_r", "joint1-r", "r1"}},
    {{"joint2-a1_r", "joint2-r", "r2"}},
    {{"joint3-a1_r", "joint3-r", "r3"}},
    {{"joint4-a1_r", "joint4-r", "r4"}},
    {{"joint5-a1_r", "joint5-r", "r5"}},
    {{"joint6-a1_r", "joint6-r", "r6"}},
    {{"joint7-a1_r", "joint7-r", "r7"}},
}};

enum class FsmState {
    Passive,
    Damping,
    FixStand,
    Ready,
    Test,
    TableTennis,
};

std::string stateName(FsmState state) {
    switch (state) {
        case FsmState::Passive:
            return "PASSIVE";
        case FsmState::Damping:
            return "DAMPING";
        case FsmState::FixStand:
            return "FIXSTAND";
        case FsmState::Ready:
            return "READY";
        case FsmState::Test:
            return "TEST";
        case FsmState::TableTennis:
            return "TABLE_TENNIS";
    }
    return "UNKNOWN";
}

std::array<double, kNumJoints> vectorToArray(
    const std::vector<double>& values,
    const std::array<double, kNumJoints>& fallback,
    const char* name) {
    if (values.empty()) return fallback;
    if (values.size() != kNumJoints) {
        throw std::runtime_error(std::string(name) + " must contain 7 values");
    }
    std::array<double, kNumJoints> out{};
    std::copy(values.begin(), values.end(), out.begin());
    return out;
}

std::array<double, kNumJoints> clampQ(
    const std::array<double, kNumJoints>& q,
    const std::array<double, kNumJoints>& q_min,
    const std::array<double, kNumJoints>& q_max) {
    std::array<double, kNumJoints> out{};
    for (size_t i = 0; i < kNumJoints; ++i) {
        out[i] = std::clamp(q[i], q_min[i], q_max[i]);
    }
    return out;
}

std::optional<std::array<double, kNumJoints>> orderedJointVector(
    const std::vector<std::string>& names,
    const std::vector<double>& values,
    const std::optional<std::array<double, kNumJoints>>& fallback) {
    if (values.empty()) return std::nullopt;
    if (names.empty() && values.size() >= kNumJoints) {
        std::array<double, kNumJoints> out{};
        std::copy_n(values.begin(), kNumJoints, out.begin());
        return out;
    }

    std::unordered_map<std::string, size_t> index;
    for (size_t i = 0; i < names.size(); ++i) index[names[i]] = i;
    std::array<double, kNumJoints> out = fallback.value_or(std::array<double, kNumJoints>{});
    int found = 0;
    for (size_t j = 0; j < kRightJointAliases.size(); ++j) {
        for (const char* alias : kRightJointAliases[j]) {
            auto it = index.find(alias);
            if (it != index.end() && it->second < values.size()) {
                out[j] = values[it->second];
                ++found;
                break;
            }
        }
    }
    if (found == static_cast<int>(kNumJoints)) return out;
    if (!fallback.has_value() && values.size() >= kNumJoints) {
        std::copy_n(values.begin(), kNumJoints, out.begin());
        return out;
    }
    if (found > 0 && fallback.has_value()) return out;
    return std::nullopt;
}

std::string vecToString(const std::array<double, kNumJoints>& q, int precision = 3) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(precision);
    os << "[";
    for (size_t i = 0; i < q.size(); ++i) {
        if (i) os << ", ";
        os << q[i];
    }
    os << "]";
    return os.str();
}

std::string lowerTrim(std::string s) {
    auto is_space = [](unsigned char c) { return std::isspace(c) != 0; };
    s.erase(s.begin(), std::find_if_not(s.begin(), s.end(), is_space));
    s.erase(std::find_if_not(s.rbegin(), s.rend(), is_space).base(), s.end());
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s;
}

double maxAbsError(
    const std::array<double, kNumJoints>& a,
    const std::array<double, kNumJoints>& b) {
    double err = 0.0;
    for (size_t i = 0; i < kNumJoints; ++i) {
        err = std::max(err, std::abs(a[i] - b[i]));
    }
    return err;
}

double smoothstep(double x) {
    x = std::clamp(x, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

double smoothEnvelope(double t, double active_s, double ramp_s) {
    if (active_s <= 0.0) return 0.0;
    const double ramp = std::min(std::max(ramp_s, 0.0), active_s * 0.5);
    if (ramp <= 1e-9) return 1.0;

    if (t < ramp) {
        return smoothstep(t / ramp);
    }
    if (t > active_s - ramp) {
        return smoothstep((active_s - t) / ramp);
    }
    return 1.0;
}

}  // namespace

class A1TTFsmSupervisor : public rclcpp::Node {
public:
    A1TTFsmSupervisor() : Node("a1_tt_fsm_supervisor") {
        control_hz_ = declare_parameter<double>("control_hz", 50.0);
        joint_timeout_s_ = declare_parameter<double>("joint_timeout_s", 2.0);
        ready_tolerance_rad_ = declare_parameter<double>("ready_tolerance_rad", 0.08);
        fixstand_timeout_s_ = declare_parameter<double>("fixstand_timeout_s", 10.0);
        fixstand_interp_s_ = declare_parameter<double>("fixstand_interp_s", 2.0);
        fixstand_enable_settle_s_ = declare_parameter<double>("fixstand_enable_settle_s", 0.50);
        enable_republish_ticks_ = declare_parameter<int>("enable_republish_ticks", 0);
        require_ready_for_tt_ = declare_parameter<bool>("require_ready_for_tt", true);
        hold_ready_ = declare_parameter<bool>("hold_ready", true);
        diag_every_ = declare_parameter<int>("diag_every", 50);
        test_enabled_ = declare_parameter<bool>("test_enabled", false);
        require_ready_for_test_ = declare_parameter<bool>("require_ready_for_test", true);
        test_signal_type_ = lowerTrim(declare_parameter<std::string>("test_signal_type", "sine"));
        test_joint_index_ = declare_parameter<int>("test_joint_index", 1);
        test_freq_hz_ = declare_parameter<double>("test_freq_hz", 0.5);
        test_chirp_start_hz_ = declare_parameter<double>("test_chirp_start_hz", 0.1);
        test_chirp_end_hz_ = declare_parameter<double>("test_chirp_end_hz", 3.0);
        test_amplitude_rad_ = declare_parameter<double>("test_amplitude_rad", 0.12);
        test_cycles_ = declare_parameter<double>("test_cycles", 8.0);
        test_duration_s_ = declare_parameter<double>("test_duration_s", 0.0);
        test_warmup_s_ = declare_parameter<double>("test_warmup_s", 2.0);
        test_post_hold_s_ = declare_parameter<double>("test_post_hold_s", 1.0);
        test_ramp_s_ = declare_parameter<double>("test_ramp_s", 0.5);
        test_done_to_ready_ = declare_parameter<bool>("test_done_to_ready", true);

        joystick_topic_ = declare_parameter<std::string>("joystick_topic", "/joystick_info");
        command_topic_ = declare_parameter<std::string>("command_topic", "/a1_tt/fsm_command");
        state_topic_ = declare_parameter<std::string>("state_topic", "/a1_tt/fsm_state");
        joint_state_topic_ = declare_parameter<std::string>("joint_state_topic", "/right_joint_states");
        action_topic_ = declare_parameter<std::string>("action_topic", "/model_action");
        right_movej_topic_ = declare_parameter<std::string>("right_movej_topic", "/movej_right_angle");
        enable_topic_ = declare_parameter<std::string>("enable_topic", "/model_control/enable");
        policy_enable_topic_ = declare_parameter<std::string>("policy_enable_topic", "/a1_tt/policy_enable");
        damping_topic_ = declare_parameter<std::string>("damping_topic", "/model_control/damping");
        fixstand_use_movej_ = declare_parameter<bool>("fixstand_use_movej", false);
        damping_exit_delay_s_ = declare_parameter<double>("damping_exit_delay_s", 0.10);

        joystick_fixstand_code_ = declare_parameter<int>("joystick_fixstand_code", 27);
        joystick_table_tennis_code_ = declare_parameter<int>("joystick_table_tennis_code", 28);
        joystick_passive_code_ = declare_parameter<int>("joystick_passive_code", 21);
        joystick_debounce_s_ = declare_parameter<double>("joystick_debounce_s", 0.50);

        default_q_ = vectorToArray(
            declare_parameter<std::vector<double>>(
                "default_q", std::vector<double>(kDefaultRightQ.begin(), kDefaultRightQ.end())),
            kDefaultRightQ,
            "default_q");
        q_min_ = vectorToArray(
            declare_parameter<std::vector<double>>(
                "q_min", std::vector<double>(kRightQMin.begin(), kRightQMin.end())),
            kRightQMin,
            "q_min");
        q_max_ = vectorToArray(
            declare_parameter<std::vector<double>>(
                "q_max", std::vector<double>(kRightQMax.begin(), kRightQMax.end())),
            kRightQMax,
            "q_max");
        default_q_ = clampQ(default_q_, q_min_, q_max_);
        test_joint_index_ = std::clamp(test_joint_index_, 1, static_cast<int>(kNumJoints));
        if (test_signal_type_ != "sine" && test_signal_type_ != "chirp"
            && test_signal_type_ != "step") {
            RCLCPP_WARN(
                get_logger(),
                "unknown test_signal_type '%s', falling back to sine",
                test_signal_type_.c_str());
            test_signal_type_ = "sine";
        }
        test_freq_hz_ = std::max(test_freq_hz_, 1e-6);
        test_chirp_start_hz_ = std::max(test_chirp_start_hz_, 1e-6);
        test_chirp_end_hz_ = std::max(test_chirp_end_hz_, 1e-6);
        test_amplitude_rad_ = std::max(test_amplitude_rad_, 0.0);
        test_cycles_ = std::max(test_cycles_, 0.0);
        test_duration_s_ = std::max(test_duration_s_, 0.0);
        test_warmup_s_ = std::max(test_warmup_s_, 0.0);
        test_post_hold_s_ = std::max(test_post_hold_s_, 0.0);
        test_ramp_s_ = std::max(test_ramp_s_, 0.0);
        fixstand_enable_settle_s_ = std::max(fixstand_enable_settle_s_, 0.0);

        action_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(action_topic_, 10);
        right_movej_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(right_movej_topic_, 10);
        enable_pub_ = create_publisher<std_msgs::msg::Bool>(enable_topic_, 10);
        policy_enable_pub_ = create_publisher<std_msgs::msg::Bool>(policy_enable_topic_, 10);
        damping_pub_ = create_publisher<std_msgs::msg::Bool>(damping_topic_, 10);
        state_pub_ = create_publisher<std_msgs::msg::String>(state_topic_, 10);

        joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
            joint_state_topic_, 10,
            std::bind(&A1TTFsmSupervisor::jointCb, this, std::placeholders::_1));
        joystick_sub_ = create_subscription<std_msgs::msg::Int32>(
            joystick_topic_, 10,
            std::bind(&A1TTFsmSupervisor::joystickCb, this, std::placeholders::_1));
        command_sub_ = create_subscription<std_msgs::msg::String>(
            command_topic_, 10,
            std::bind(&A1TTFsmSupervisor::commandCb, this, std::placeholders::_1));

        const double period = 1.0 / std::max(control_hz_, 1e-6);
        timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::duration<double>(period)),
            std::bind(&A1TTFsmSupervisor::tick, this));

        enterPassive("startup");
        RCLCPP_INFO(
            get_logger(),
            "a1_tt_fsm_supervisor ready: fixstand=%d play=%d passive=%d damping_topic=%s default_q=%s fixstand_interp=%.3f fixstand_enable_settle=%.3f test_enabled=%s test_joint=%d test_freq=%.3f test_amp=%.3f",
            joystick_fixstand_code_,
            joystick_table_tennis_code_,
            joystick_passive_code_,
            damping_topic_.c_str(),
            vecToString(default_q_).c_str(),
            fixstand_interp_s_,
            fixstand_enable_settle_s_,
            test_enabled_ ? "true" : "false",
            test_joint_index_,
            test_freq_hz_,
            test_amplitude_rad_);
    }

private:
    double nowSec() const {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    bool jointStale() const {
        return !last_joint_time_.has_value() || nowSec() - last_joint_time_.value() > joint_timeout_s_;
    }

    void jointCb(const sensor_msgs::msg::JointState::SharedPtr msg) {
        auto q = orderedJointVector(msg->name, msg->position, q_);
        if (!q.has_value()) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "joint state did not contain 7 usable right-arm positions");
            return;
        }
        q_ = q.value();
        last_joint_time_ = nowSec();
        ++joint_rx_count_;
    }

    void joystickCb(const std_msgs::msg::Int32::SharedPtr msg) {
        const int code = msg->data;
        const double t = nowSec();
        if (last_joystick_code_.has_value() && last_joystick_time_.has_value() &&
            code == last_joystick_code_.value() &&
            t - last_joystick_time_.value() < joystick_debounce_s_) {
            return;
        }
        last_joystick_code_ = code;
        last_joystick_time_ = t;

        if (code == joystick_passive_code_) {
            if (state_ == FsmState::Damping) return;
            enterDamping("joystick_damping");
        } else if (code == joystick_fixstand_code_) {
            if (state_ == FsmState::FixStand || state_ == FsmState::Ready) return;
            startFixStand("joystick_fixstand");
        } else if (code == joystick_table_tennis_code_) {
            if (test_enabled_) {
                if (state_ == FsmState::Test) return;
                enterTest("joystick_test");
            } else {
                if (state_ == FsmState::TableTennis) return;
                enterTableTennis("joystick_table_tennis");
            }
        }
    }

    void commandCb(const std_msgs::msg::String::SharedPtr msg) {
        const std::string command = lowerTrim(msg->data);
        if (command == "damping" || command == "damp" || command == "back" ||
            command == "stop" || command == "safe" || command == "0") {
            enterDamping("command_" + command);
        } else if (command == "passive" || command == "disable" || command == "off" ||
                   command == "servo_off") {
            enterPassive("command_" + command);
        } else if (command == "fixstand" || command == "default" || command == "ready" ||
                   command == "stand" || command == "1") {
            startFixStand("command_" + command);
        } else if (command == "test" || command == "joint_id" || command == "id") {
            enterTest("command_" + command);
        } else if (command == "tt" || command == "table_tennis" || command == "pingpong" ||
                   command == "play" || command == "2") {
            if (test_enabled_) {
                enterTest("command_test_mapped_from_" + command);
            } else {
                enterTableTennis("command_" + command);
            }
        } else if (command == "force_tt" || command == "real_tt") {
            enterTableTennis("command_" + command);
        } else {
            RCLCPP_WARN(get_logger(), "unknown FSM command: '%s'", msg->data.c_str());
        }
    }

    void tick() {
        ++tick_;
        if (state_ != FsmState::Passive && state_ != FsmState::Damping && jointStale()) {
            enterPassive("stale_joint_state");
        }

        const bool force_enable_publish =
            enable_republish_ticks_ > 0 && tick_ % enable_republish_ticks_ == 0;
        switch (state_) {
            case FsmState::Passive:
                publishPolicyEnable(false, force_enable_publish);
                publishServoEnable(false, force_enable_publish);
                publishDamping(false, force_enable_publish);
                break;
            case FsmState::Damping:
                tickDamping(force_enable_publish);
                break;
            case FsmState::FixStand:
                tickFixStand(force_enable_publish);
                break;
            case FsmState::Ready:
                publishPolicyEnable(false, force_enable_publish);
                publishDamping(false, force_enable_publish);
                publishServoEnable(true, force_enable_publish);
                if (hold_ready_) publishAction(default_q_);
                break;
            case FsmState::Test:
                tickTest(force_enable_publish);
                break;
            case FsmState::TableTennis:
                publishPolicyEnable(true, force_enable_publish);
                publishDamping(false, force_enable_publish);
                publishServoEnable(true, force_enable_publish);
                break;
        }

        if (diag_every_ > 0 && tick_ % diag_every_ == 0) {
            RCLCPP_INFO(
                get_logger(),
                "state=%s reason=%s joint_age=%.3f joint_rx=%lu q=%s",
                stateName(state_).c_str(),
                last_reason_.c_str(),
                last_joint_time_.has_value() ? nowSec() - last_joint_time_.value() : std::numeric_limits<double>::infinity(),
                joint_rx_count_,
                q_.has_value() ? vecToString(q_.value()).c_str() : "(no joint state)");
        }
        publishState();
    }

    void tickDamping(bool force_enable_publish) {
        publishPolicyEnable(false, force_enable_publish);
        publishServoEnable(false, force_enable_publish);
        publishDamping(true, force_enable_publish);
    }

    void tickFixStand(bool force_enable_publish) {
        publishPolicyEnable(false, force_enable_publish);
        publishDamping(false, force_enable_publish);
        if (!fixstand_cmd_q_.has_value()) {
            fixstand_cmd_q_ = q_.value_or(default_q_);
        }
        if (!fixstand_start_q_.has_value()) {
            fixstand_start_q_ = fixstand_cmd_q_.value();
        }

        if (!dampingExitReady()) {
            publishServoEnable(false, force_enable_publish);
            if (!fixstand_use_movej_) {
                publishAction(fixstand_start_q_.value());
            }
            return;
        }
        if (fixstand_use_movej_) {
            publishServoEnable(false, force_enable_publish);
            if (!fixstand_movej_sent_) {
                publishMoveJ(default_q_);
                fixstand_movej_sent_ = true;
            }
        } else {
            const double now = nowSec();
            if (!fixstand_enable_start_time_.has_value()) {
                fixstand_enable_start_time_ = now;
            }
            const bool settling =
                now - fixstand_enable_start_time_.value() < fixstand_enable_settle_s_;
            publishServoEnable(true, settling || force_enable_publish);
            if (settling) {
                publishAction(fixstand_start_q_.value());
                return;
            }
            if (!fixstand_start_time_.has_value()) {
                fixstand_start_time_ = now;
            }
        }
        const double elapsed = fixstand_start_time_.has_value()
            ? nowSec() - fixstand_start_time_.value()
            : fixstand_interp_s_;
        const double alpha = fixstand_interp_s_ <= 0.0
            ? 1.0
            : std::clamp(elapsed / fixstand_interp_s_, 0.0, 1.0);
        std::array<double, kNumJoints> cmd{};
        for (size_t i = 0; i < kNumJoints; ++i) {
            cmd[i] = fixstand_start_q_.value()[i] +
                (default_q_[i] - fixstand_start_q_.value()[i]) * alpha;
        }
        fixstand_cmd_q_ = clampQ(cmd, q_min_, q_max_);
        if (!fixstand_use_movej_) {
            publishAction(fixstand_cmd_q_.value());
        }

        if (q_.has_value() && maxAbsError(q_.value(), default_q_) <= ready_tolerance_rad_) {
            enterReady("fixstand_reached");
            return;
        }

        const double total_elapsed = fixstand_enter_time_.has_value()
            ? nowSec() - fixstand_enter_time_.value()
            : elapsed;
        if (fixstand_timeout_s_ > 0.0 && total_elapsed > fixstand_timeout_s_) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "FIXSTAND timeout; holding default target, q_error=%.3f total_elapsed=%.3f",
                q_.has_value() ? maxAbsError(q_.value(), default_q_) : std::numeric_limits<double>::infinity(),
                total_elapsed);
        }
    }

    double testActiveS() const {
        if (test_duration_s_ > 0.0) return test_duration_s_;
        return test_freq_hz_ > 1e-9 ? test_cycles_ / test_freq_hz_ : 0.0;
    }

    double testTotalS() const {
        return test_warmup_s_ + testActiveS() + test_post_hold_s_;
    }

    std::string testPhase(double elapsed) const {
        if (elapsed < test_warmup_s_) return "warmup";
        if (elapsed < test_warmup_s_ + testActiveS()) return test_signal_type_;
        return "post_hold";
    }

    double testInstantFreqHz(double t_active, double active_s) const {
        if (test_signal_type_ != "chirp" || active_s <= 1e-9) return test_freq_hz_;
        const double alpha = std::clamp(t_active / active_s, 0.0, 1.0);
        return test_chirp_start_hz_ + (test_chirp_end_hz_ - test_chirp_start_hz_) * alpha;
    }

    double testPhaseRad(double t_active, double active_s) const {
        if (test_signal_type_ == "chirp" && active_s > 1e-9) {
            const double k = (test_chirp_end_hz_ - test_chirp_start_hz_) / active_s;
            return 2.0 * kPi * (test_chirp_start_hz_ * t_active + 0.5 * k * t_active * t_active);
        }
        return 2.0 * kPi * test_freq_hz_ * t_active;
    }

    void tickTest(bool force_enable_publish) {
        publishPolicyEnable(false, force_enable_publish);
        publishDamping(false, force_enable_publish);
        publishServoEnable(true, force_enable_publish);

        const double elapsed = test_start_time_.has_value()
            ? nowSec() - test_start_time_.value()
            : testTotalS();
        const double active_s = testActiveS();
        if (elapsed > testTotalS()) {
            publishAction(default_q_);
            if (test_done_to_ready_) {
                enterReady("test_done");
            } else {
                enterPassive("test_done");
            }
            return;
        }

        std::array<double, kNumJoints> q = default_q_;
        if (elapsed >= test_warmup_s_ && elapsed < test_warmup_s_ + active_s) {
            const double t_active = elapsed - test_warmup_s_;
            const double env = smoothEnvelope(t_active, active_s, test_ramp_s_);
            const double phase = testPhaseRad(t_active, active_s);
            const int joint = std::clamp(test_joint_index_, 1, static_cast<int>(kNumJoints)) - 1;
            const double wave = (test_signal_type_ == "step")
                ? ((std::sin(phase) >= 0.0) ? 1.0 : -1.0)
                : std::sin(phase);
            q[joint] = default_q_[joint] + test_amplitude_rad_ * env * wave;
            q = clampQ(q, q_min_, q_max_);
        }
        publishAction(q);
    }

    void startFixStand(const std::string& reason) {
        if (jointStale()) {
            RCLCPP_WARN(
                get_logger(),
                "ignore FixStand command (%s): no fresh joint state on %s",
                reason.c_str(),
                joint_state_topic_.c_str());
            enterPassive("fixstand_no_joint_state");
            return;
        }
        state_ = FsmState::FixStand;
        last_reason_ = reason;
        fixstand_cmd_q_ = clampQ(q_.value(), q_min_, q_max_);
        fixstand_start_q_ = fixstand_cmd_q_.value();
        fixstand_enter_time_ = nowSec();
        fixstand_enable_start_time_.reset();
        fixstand_start_time_.reset();
        fixstand_movej_sent_ = false;
        test_start_time_.reset();
        // The bottom controller's damping latch outlives this supervisor.
        // After a local FSM restart our state starts at PASSIVE even when the
        // robot is still damped.  Always publish damping=false first and wait
        // before MoveJ; otherwise armcontrol drops the only MoveJ command as
        // "damping active" and the arm never leaves its current pose.
        startDampingExitGuard(true);
        publishDamping(false, true);
        publishPolicyEnable(false, true);
        if (fixstand_use_movej_) {
            publishServoEnable(false, true);
            if (dampingExitReady()) {
                publishMoveJ(default_q_);
                fixstand_movej_sent_ = true;
            }
        } else {
            if (dampingExitReady()) {
                publishServoEnable(true, true);
                publishAction(fixstand_cmd_q_.value());
                fixstand_enable_start_time_ = nowSec();
            } else {
                publishServoEnable(false, true);
                publishAction(fixstand_cmd_q_.value());
            }
        }
        publishState();
        RCLCPP_INFO(
            get_logger(),
            "enter FIXSTAND: reason=%s start=%s target=%s interp_s=%.3f enable_settle_s=%.3f mode=%s damping_exit_delay=%.3f",
            reason.c_str(),
            vecToString(fixstand_cmd_q_.value()).c_str(),
            vecToString(default_q_).c_str(),
            fixstand_interp_s_,
            fixstand_enable_settle_s_,
            fixstand_use_movej_ ? "movej" : "model_action",
            damping_exit_guard_until_.has_value() ? damping_exit_delay_s_ : 0.0);
    }

    void enterReady(const std::string& reason) {
        state_ = FsmState::Ready;
        last_reason_ = reason;
        fixstand_cmd_q_.reset();
        fixstand_start_q_.reset();
        fixstand_enter_time_.reset();
        fixstand_enable_start_time_.reset();
        fixstand_start_time_.reset();
        fixstand_movej_sent_ = false;
        test_start_time_.reset();
        clearDampingExitGuard();
        publishDamping(false, true);
        publishPolicyEnable(false, true);
        publishServoEnable(true, true);
        if (hold_ready_) publishAction(default_q_);
        publishState();
        RCLCPP_INFO(get_logger(), "enter READY: reason=%s", reason.c_str());
    }

    void enterTest(const std::string& reason) {
        if (require_ready_for_test_ && state_ != FsmState::Ready) {
            RCLCPP_WARN(
                get_logger(),
                "ignore TEST command (%s): current state is %s, run FixStand first",
                reason.c_str(),
                stateName(state_).c_str());
            return;
        }
        if (jointStale()) {
            enterPassive("test_no_joint_state");
            return;
        }
        if (testActiveS() <= 0.0) {
            RCLCPP_WARN(
                get_logger(),
                "ignore TEST command (%s): invalid active duration, signal=%s freq=%.6f cycles=%.3f duration=%.3f",
                reason.c_str(),
                test_signal_type_.c_str(),
                test_freq_hz_,
                test_cycles_,
                test_duration_s_);
            return;
        }
        state_ = FsmState::Test;
        last_reason_ = reason;
        fixstand_cmd_q_.reset();
        fixstand_start_q_.reset();
        fixstand_enter_time_.reset();
        fixstand_enable_start_time_.reset();
        fixstand_start_time_.reset();
        fixstand_movej_sent_ = false;
        test_start_time_ = nowSec();
        clearDampingExitGuard();
        publishDamping(false, true);
        publishPolicyEnable(false, true);
        publishServoEnable(true, true);
        publishAction(default_q_);
        publishState();
        RCLCPP_INFO(
            get_logger(),
            "enter TEST: reason=%s joint=%d center=%s signal=%s freq=%.3f chirp=[%.3f, %.3f] amp=%.3f active_s=%.3f cycles=%.3f warmup=%.3f ramp=%.3f post=%.3f action_format=position",
            reason.c_str(),
            test_joint_index_,
            vecToString(default_q_).c_str(),
            test_signal_type_.c_str(),
            test_freq_hz_,
            test_chirp_start_hz_,
            test_chirp_end_hz_,
            test_amplitude_rad_,
            testActiveS(),
            test_cycles_,
            test_warmup_s_,
            test_ramp_s_,
            test_post_hold_s_);
    }

    void enterTableTennis(const std::string& reason) {
        if (require_ready_for_tt_ && state_ != FsmState::Ready) {
            RCLCPP_WARN(
                get_logger(),
                "ignore TableTennis command (%s): current state is %s, run FixStand first",
                reason.c_str(),
                stateName(state_).c_str());
            return;
        }
        if (jointStale()) {
            enterPassive("table_tennis_no_joint_state");
            return;
        }
        state_ = FsmState::TableTennis;
        last_reason_ = reason;
        fixstand_cmd_q_.reset();
        fixstand_start_q_.reset();
        fixstand_enter_time_.reset();
        fixstand_enable_start_time_.reset();
        fixstand_start_time_.reset();
        fixstand_movej_sent_ = false;
        test_start_time_.reset();
        clearDampingExitGuard();
        publishDamping(false, true);
        publishPolicyEnable(true, true);
        // READY already enabled the SDK servo. Re-publishing true makes the
        // bottom controller re-enable seven motors synchronously (7 x 100 ms),
        // stalling the 100 Hz command loop just as policy control starts.
        publishServoEnable(true, false);
        publishState();
        RCLCPP_INFO(get_logger(), "enter TABLE_TENNIS: reason=%s", reason.c_str());
    }

    void enterDamping(const std::string& reason) {
        if (state_ == FsmState::FixStand && fixstand_use_movej_) {
            RCLCPP_WARN(
                get_logger(),
                "ignore DAMPING command (%s): FixStand movej is active; bottom controller refuses damping during movej",
                reason.c_str());
            return;
        }
        state_ = FsmState::Damping;
        last_reason_ = reason;
        fixstand_cmd_q_.reset();
        fixstand_start_q_.reset();
        fixstand_enter_time_.reset();
        fixstand_enable_start_time_.reset();
        fixstand_start_time_.reset();
        fixstand_movej_sent_ = false;
        test_start_time_.reset();
        clearDampingExitGuard();
        publishPolicyEnable(false, true);
        publishServoEnable(false, true);
        publishDamping(true, true);
        publishState();
        RCLCPP_INFO(get_logger(), "enter DAMPING: reason=%s", reason.c_str());
    }

    void enterPassive(const std::string& reason) {
        state_ = FsmState::Passive;
        last_reason_ = reason;
        fixstand_cmd_q_.reset();
        fixstand_start_q_.reset();
        fixstand_enter_time_.reset();
        fixstand_enable_start_time_.reset();
        fixstand_start_time_.reset();
        fixstand_movej_sent_ = false;
        test_start_time_.reset();
        clearDampingExitGuard();
        publishDamping(false, true);
        publishPolicyEnable(false, true);
        publishServoEnable(false, true);
        publishState();
        RCLCPP_INFO(get_logger(), "enter PASSIVE: reason=%s", reason.c_str());
    }

    void publishAction(const std::array<double, kNumJoints>& q) {
        std_msgs::msg::Float64MultiArray msg;
        msg.data.assign(q.begin(), q.end());
        action_pub_->publish(msg);
    }

    void publishMoveJ(const std::array<double, kNumJoints>& q) {
        std_msgs::msg::Float64MultiArray msg;
        msg.data.assign(q.begin(), q.end());
        right_movej_pub_->publish(msg);
    }

    void startDampingExitGuard(bool was_damping) {
        if (was_damping && damping_exit_delay_s_ > 0.0) {
            damping_exit_guard_until_ = nowSec() + damping_exit_delay_s_;
        } else {
            clearDampingExitGuard();
        }
    }

    void clearDampingExitGuard() {
        damping_exit_guard_until_.reset();
    }

    bool dampingExitReady() const {
        return !damping_exit_guard_until_.has_value() ||
            nowSec() >= damping_exit_guard_until_.value();
    }

    void publishServoEnable(bool enabled, bool force = false) {
        // armcontrol re-syncs command targets to the current measured joints
        // on every true enable message. Re-sending true while READY/TT can
        // erase the intended hold target if the arm sags between packets.
        // Force is used only for explicit resync windows such as FixStand
        // enable-settle, where the FSM is still holding current position.
        if (!force && last_servo_enable_.has_value() &&
            last_servo_enable_.value() == enabled) {
            return;
        }
        last_servo_enable_ = enabled;
        std_msgs::msg::Bool msg;
        msg.data = enabled;
        enable_pub_->publish(msg);
    }

    void publishPolicyEnable(bool enabled, bool force = false) {
        if (!force && last_policy_enable_.has_value() &&
            last_policy_enable_.value() == enabled) {
            return;
        }
        last_policy_enable_ = enabled;
        std_msgs::msg::Bool msg;
        msg.data = enabled;
        policy_enable_pub_->publish(msg);
    }

    void publishDamping(bool enabled, bool force = false) {
        if (!force && last_damping_.has_value() && last_damping_.value() == enabled) {
            return;
        }
        last_damping_ = enabled;
        std_msgs::msg::Bool msg;
        msg.data = enabled;
        damping_pub_->publish(msg);
    }

    void publishState() {
        std_msgs::msg::String msg;
        std::ostringstream os;
        os << "state=" << stateName(state_)
           << " reason=" << last_reason_
           << " ready_tolerance_rad=" << ready_tolerance_rad_;
        os << " joint_age_s=";
        if (last_joint_time_.has_value()) {
            os << nowSec() - last_joint_time_.value();
        } else {
            os << "inf";
        }
        os << " joint_rx_count=" << joint_rx_count_;
        if (q_.has_value()) {
            os << " q_error=" << maxAbsError(q_.value(), default_q_);
        } else {
            os << " q_error=nan";
        }
        if (state_ == FsmState::Test) {
            const double elapsed = test_start_time_.has_value()
                ? nowSec() - test_start_time_.value()
                : 0.0;
            os << " test_joint=" << test_joint_index_
               << " test_signal_type=" << test_signal_type_
               << " test_freq_hz=" << test_freq_hz_
               << " test_chirp_start_hz=" << test_chirp_start_hz_
               << " test_chirp_end_hz=" << test_chirp_end_hz_
               << " test_amplitude_rad=" << test_amplitude_rad_
               << " test_phase=" << testPhase(elapsed)
               << " test_instant_freq_hz="
               << testInstantFreqHz(std::max(0.0, elapsed - test_warmup_s_), testActiveS())
               << " test_elapsed_s=" << elapsed
               << " test_total_s=" << testTotalS();
        } else if (state_ == FsmState::FixStand) {
            const double now = nowSec();
            const double total_elapsed = fixstand_enter_time_.has_value()
                ? now - fixstand_enter_time_.value()
                : 0.0;
            const double ramp_elapsed = fixstand_start_time_.has_value()
                ? now - fixstand_start_time_.value()
                : 0.0;
            const char* phase = "damping_exit";
            if (dampingExitReady()) {
                if (!fixstand_start_time_.has_value()) {
                    phase = "enable_settle";
                } else {
                    phase = "ramp";
                }
            }
            os << " fixstand_phase=" << phase
               << " fixstand_total_elapsed_s=" << total_elapsed
               << " fixstand_ramp_elapsed_s=" << ramp_elapsed
               << " fixstand_enable_settle_s=" << fixstand_enable_settle_s_
               << " fixstand_interp_s=" << fixstand_interp_s_;
        }
        msg.data = os.str();
        state_pub_->publish(msg);
    }

    FsmState state_ = FsmState::Passive;
    std::string last_reason_ = "startup";
    double control_hz_ = 50.0;
    double joint_timeout_s_ = 2.0;
    double ready_tolerance_rad_ = 0.08;
    double fixstand_timeout_s_ = 10.0;
    double fixstand_interp_s_ = 2.0;
    double fixstand_enable_settle_s_ = 0.50;
    int enable_republish_ticks_ = 0;
    bool require_ready_for_tt_ = true;
    bool hold_ready_ = true;
    bool fixstand_use_movej_ = false;
    double damping_exit_delay_s_ = 0.10;
    int diag_every_ = 50;
    bool test_enabled_ = false;
    bool require_ready_for_test_ = true;
    std::string test_signal_type_ = "sine";
    int test_joint_index_ = 1;
    double test_freq_hz_ = 0.5;
    double test_chirp_start_hz_ = 0.1;
    double test_chirp_end_hz_ = 3.0;
    double test_amplitude_rad_ = 0.12;
    double test_cycles_ = 8.0;
    double test_duration_s_ = 0.0;
    double test_warmup_s_ = 2.0;
    double test_post_hold_s_ = 1.0;
    double test_ramp_s_ = 0.5;
    bool test_done_to_ready_ = true;

    std::string joystick_topic_;
    std::string command_topic_;
    std::string state_topic_;
    std::string joint_state_topic_;
    std::string action_topic_;
    std::string right_movej_topic_;
    std::string enable_topic_;
    std::string policy_enable_topic_;
    std::string damping_topic_;
    int joystick_fixstand_code_ = 27;
    int joystick_table_tennis_code_ = 28;
    int joystick_passive_code_ = 21;
    double joystick_debounce_s_ = 0.50;
    std::optional<int> last_joystick_code_;
    std::optional<double> last_joystick_time_;

    std::array<double, kNumJoints> default_q_ = kDefaultRightQ;
    std::array<double, kNumJoints> q_min_ = kRightQMin;
    std::array<double, kNumJoints> q_max_ = kRightQMax;
    std::optional<std::array<double, kNumJoints>> q_;
    std::optional<std::array<double, kNumJoints>> fixstand_cmd_q_;
    std::optional<std::array<double, kNumJoints>> fixstand_start_q_;
    std::optional<double> last_joint_time_;
    std::optional<double> fixstand_enter_time_;
    std::optional<double> fixstand_enable_start_time_;
    std::optional<double> fixstand_start_time_;
    std::optional<double> test_start_time_;
    std::optional<double> damping_exit_guard_until_;
    uint64_t joint_rx_count_ = 0;
    bool fixstand_movej_sent_ = false;
    std::optional<bool> last_servo_enable_;
    std::optional<bool> last_policy_enable_;
    std::optional<bool> last_damping_;
    long tick_ = 0;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr action_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr right_movej_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr enable_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr policy_enable_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr damping_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr state_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Subscription<std_msgs::msg::Int32>::SharedPtr joystick_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr command_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<A1TTFsmSupervisor>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        std::cerr << "a1_tt_fsm_supervisor failed: " << e.what() << std::endl;
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
