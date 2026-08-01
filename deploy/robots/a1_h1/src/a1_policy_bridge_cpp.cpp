#include <onnxruntime_cxx_api.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/bool.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_msgs/msg/string.hpp>

namespace fs = std::filesystem;

namespace {

constexpr double kActionScale = 0.25;
constexpr double kClipActions = 10.0;
constexpr double kClipObs = 100.0;
constexpr double kSoftJointLimitFactor = 0.95;
constexpr int kFrameSize = 39;
constexpr int kHistory = 5;
constexpr int kObsSize = kFrameSize * kHistory;
constexpr double kHitPlaneX = -1.60;
constexpr double kHomeY = 0.76;
constexpr double kPaddleYOffset = -0.66;
constexpr double kHitBodyHeight = 0.028;
constexpr double kGravity = 9.81;
constexpr double kZBounce = 0.78;

const std::array<std::string, 7> kRightJointNames = {
    "joint1-a1_r", "joint2-a1_r", "joint3-a1_r", "joint4-a1_r",
    "joint5-a1_r", "joint6-a1_r", "joint7-a1_r"};

const std::array<std::array<const char*, 3>, 7> kRightJointAliases = {{
    {{"joint1-a1_r", "joint1-r", "r1"}},
    {{"joint2-a1_r", "joint2-r", "r2"}},
    {{"joint3-a1_r", "joint3-r", "r3"}},
    {{"joint4-a1_r", "joint4-r", "r4"}},
    {{"joint5-a1_r", "joint5-r", "r5"}},
    {{"joint6-a1_r", "joint6-r", "r6"}},
    {{"joint7-a1_r", "joint7-r", "r7"}},
}};

const std::array<double, 7> kDefaultRightQ = {
    0.569, -0.692, 0.717, 1.13, -1.24, 0.0314, 0.772};
const std::array<double, 7> kRightQMin = {
    -1.05, -3.14, -2.76, -1.92, -2.76, -1.57, -2.76};
const std::array<double, 7> kRightQMax = {
    3.14, 0.262, 2.76, 1.92, 2.76, 1.57, 2.76};
const std::array<float, 3> kRobotTablePos = {-1.8f, 0.76f, 0.0282f};
const std::array<float, 3> kPredSentinel = {
    static_cast<float>(kHitPlaneX),
    static_cast<float>(kHomeY + kPaddleYOffset),
    static_cast<float>(kHitBodyHeight + 0.2)};
const std::array<double, 7> kIsaacServoVelocityLimit = {
    1.0, 1.2, 1.8, 1.6, 4.0, 3.2, 8.0};

// v8+ training first-order low-pass tau (a1_tt_config.py A1_REAL_DEPLOY_LOWPASS_TAU_S).
// Per-joint, NOT uniform: underdamped proximal r1/r2/r4 need large tau to suppress
// resonance; wrist r5/r6 stay small for swing response.
const std::array<double, 7> kIsaacServoTauS = {
    0.10, 0.10, 0.08, 0.10, 0.05, 0.05, 0.10};

fs::path findLgyRootFrom(fs::path start) {
    start = fs::absolute(start);
    if (fs::is_regular_file(start)) start = start.parent_path();
    for (fs::path cur = start; !cur.empty(); cur = cur.parent_path()) {
        if (fs::exists(cur / "Pingpong_TTRL") && fs::exists(cur / "unitree_rl_lab")) {
            return cur;
        }
        if (cur == cur.root_path()) break;
    }
    return {};
}

fs::path findLgyRoot() {
    if (const char* env = std::getenv("LGY_ROOT")) {
        fs::path root(env);
        if (fs::exists(root / "Pingpong_TTRL") && fs::exists(root / "unitree_rl_lab")) {
            return root;
        }
    }
    if (auto root = findLgyRootFrom(fs::current_path()); !root.empty()) return root;
    if (auto root = findLgyRootFrom(__FILE__); !root.empty()) return root;
    return fs::current_path();
}

std::string defaultPolicyPath() {
    return (findLgyRoot() /
            "Pingpong_TTRL/logs/a1_tt_real_v7/2026-07-24_14-18-11_resume10000_range10k_hold20k/exported/policy.onnx")
        .string();
}

const std::string kDefaultPolicy = defaultPolicyPath();

double clampDouble(double v, double lo, double hi) {
    return std::max(lo, std::min(hi, v));
}

float clampFloat(float v, float lo, float hi) {
    return std::max(lo, std::min(hi, v));
}

bool finite3(const std::array<double, 3>& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

bool finite3f(const std::array<float, 3>& v) {
    return std::isfinite(v[0]) && std::isfinite(v[1]) && std::isfinite(v[2]);
}

std::string vecToString(const std::array<double, 7>& v, int precision = 3) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(precision);
    os << "[";
    for (size_t i = 0; i < v.size(); ++i) {
        if (i) os << ", ";
        os << v[i];
    }
    os << "]";
    return os.str();
}

std::string vecToString(const std::array<float, 3>& v, int precision = 3) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(precision);
    os << "[" << v[0] << ", " << v[1] << ", " << v[2] << "]";
    return os.str();
}

std::pair<std::array<double, 7>, std::array<double, 7>> softJointRanges() {
    std::array<double, 7> q_min{};
    std::array<double, 7> q_max{};
    for (size_t i = 0; i < 7; ++i) {
        const double mid = 0.5 * (kRightQMin[i] + kRightQMax[i]);
        const double half = 0.5 * (kRightQMax[i] - kRightQMin[i]) * kSoftJointLimitFactor;
        q_min[i] = mid - half;
        q_max[i] = mid + half;
    }
    return {q_min, q_max};
}

class OrtRunner {
public:
    explicit OrtRunner(const std::string& model_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "a1_policy_bridge_cpp") {
        if (!fs::exists(model_path)) {
            throw std::runtime_error("ONNX model not found: " + model_path);
        }
        session_options_.SetGraphOptimizationLevel(ORT_ENABLE_EXTENDED);
        session_ = std::make_unique<Ort::Session>(env_, model_path.c_str(), session_options_);

        for (size_t i = 0; i < session_->GetInputCount(); ++i) {
            auto name = session_->GetInputNameAllocated(i, allocator_);
            input_name_storage_.emplace_back(name.get());
            auto type_info = session_->GetInputTypeInfo(i);
            auto shape = type_info.GetTensorTypeAndShapeInfo().GetShape();
            size_t size = 1;
            for (auto& d : shape) {
                if (d < 0) d = 1;
                size *= static_cast<size_t>(d);
            }
            input_shapes_.push_back(std::move(shape));
            input_sizes_.push_back(size);
        }

        for (size_t i = 0; i < session_->GetOutputCount(); ++i) {
            auto name = session_->GetOutputNameAllocated(i, allocator_);
            output_name_storage_.emplace_back(name.get());
        }

        for (const auto& s : input_name_storage_) input_names_.push_back(s.c_str());
        for (const auto& s : output_name_storage_) output_names_.push_back(s.c_str());
        if (input_names_.empty() || output_names_.empty()) {
            throw std::runtime_error("ONNX model has no input or output: " + model_path);
        }
    }

    size_t inputSize() const { return input_sizes_.front(); }
    const std::string& inputName() const { return input_name_storage_.front(); }

    std::vector<float> runSingle(const std::vector<float>& input) {
        if (input.size() != input_sizes_.front()) {
            throw std::runtime_error(
                "bad ONNX input size for " + input_name_storage_.front() +
                ": got " + std::to_string(input.size()) +
                ", expected " + std::to_string(input_sizes_.front()));
        }
        auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
        auto tensor = Ort::Value::CreateTensor<float>(
            memory_info,
            const_cast<float*>(input.data()),
            input.size(),
            input_shapes_.front().data(),
            input_shapes_.front().size());
        auto output = session_->Run(
            Ort::RunOptions{nullptr},
            input_names_.data(),
            &tensor,
            1,
            output_names_.data(),
            1);
        auto& out0 = output.front();
        const auto count = out0.GetTensorTypeAndShapeInfo().GetElementCount();
        const float* data = out0.GetTensorData<float>();
        return std::vector<float>(data, data + count);
    }

private:
    Ort::Env env_;
    Ort::SessionOptions session_options_;
    Ort::AllocatorWithDefaultOptions allocator_;
    std::unique_ptr<Ort::Session> session_;
    std::vector<std::string> input_name_storage_;
    std::vector<std::string> output_name_storage_;
    std::vector<const char*> input_names_;
    std::vector<const char*> output_names_;
    std::vector<std::vector<int64_t>> input_shapes_;
    std::vector<size_t> input_sizes_;
};

struct BallGateConfig {
    double own_x_lo = -1.37;
    double own_x_hi = 0.0;
    double y_abs = 1.2;
    double z_min = 0.75;
    double z_max = 2.6;
    double x_max = 1.65;
    double speed_max = 15.0;
    double vx_away = -0.05;
    double hit_plane_x = kHitPlaneX;
    double behind_margin = 0.05;
    double bounce_vz_down = 0.30;
    double bounce_vz_up = 0.05;
    double bounce_near_table = 0.15;
    int confirm_frames = 1;
    int coast_frames = 1;
};

struct BallGateOutput {
    bool live = false;
    bool engaged = false;
    std::string reason = "reset";
    double first_bounce_x = std::numeric_limits<double>::quiet_NaN();
    int own_bounces = 0;
    bool hit_seen = false;
};

class BallValidityGate {
public:
    explicit BallValidityGate(BallGateConfig cfg = {}) : cfg_(cfg) {}

    void reset() {
        prev_vz_.reset();
        own_bounces_ = 0;
        hit_seen_ = false;
        live_run_ = 0;
        dead_run_ = 0;
        engaged_ = false;
        last_ = {};
    }

    BallGateOutput update(const std::array<double, 3>& pos, const std::array<double, 3>& vel) {
        updateBounces(pos, vel);
        const double first_bounce_x = predictedFirstBounceX(pos, vel);
        auto [live, reason] = classify(pos, vel, first_bounce_x);

        if (live) {
            ++live_run_;
            dead_run_ = 0;
        } else {
            ++dead_run_;
            live_run_ = 0;
        }
        if (!engaged_ && live_run_ >= std::max(1, cfg_.confirm_frames)) engaged_ = true;
        if (engaged_ && dead_run_ >= std::max(1, cfg_.coast_frames)) engaged_ = false;

        prev_vz_ = std::isfinite(vel[2]) ? std::optional<double>(vel[2]) : std::nullopt;
        last_ = {live, engaged_, reason, first_bounce_x, own_bounces_, hit_seen_};
        return last_;
    }

private:
    std::pair<bool, std::string> classify(
        const std::array<double, 3>& pos,
        const std::array<double, 3>& vel,
        double first_bounce_x) const {
        if (!finite3(pos) || !finite3(vel)) return {false, "nonfinite"};
        if (hit_seen_) return {false, "paddle_hit"};
        const double speed = std::sqrt(vel[0] * vel[0] + vel[1] * vel[1] + vel[2] * vel[2]);
        if (speed > cfg_.speed_max) return {false, "speed"};
        if (std::abs(pos[1]) > cfg_.y_abs || pos[2] > cfg_.z_max || pos[0] > cfg_.x_max) {
            return {false, "out_volume"};
        }
        if (pos[0] < cfg_.hit_plane_x - cfg_.behind_margin) return {false, "behind_hit_plane"};
        if (vel[0] > cfg_.vx_away) return {false, "moving_away"};
        if (pos[2] < cfg_.z_min) return {false, "low_or_dead"};
        if (own_bounces_ >= 2) return {false, "double_bounce"};
        if (own_bounces_ == 0) {
            if (!std::isfinite(first_bounce_x)) return {false, "no_table_bounce"};
            if (!(cfg_.own_x_lo <= first_bounce_x && first_bounce_x <= cfg_.own_x_hi)) {
                return {false, "first_bounce_out"};
            }
        }
        return {true, "valid"};
    }

    void updateBounces(const std::array<double, 3>& pos, const std::array<double, 3>& vel) {
        if (!prev_vz_.has_value() || !finite3(pos) || !finite3(vel)) return;
        const bool was_descending = prev_vz_.value() < -cfg_.bounce_vz_down;
        const bool ascending_now = vel[2] > cfg_.bounce_vz_up;
        const bool near_table = pos[2] < kZBounce + cfg_.bounce_near_table;
        const bool own_half = cfg_.own_x_lo <= pos[0] && pos[0] <= cfg_.own_x_hi;
        if (was_descending && ascending_now && near_table && own_half) {
            ++own_bounces_;
        }
    }

    static double predictedFirstBounceX(const std::array<double, 3>& pos, const std::array<double, 3>& vel) {
        if (!finite3(pos) || !finite3(vel)) return std::numeric_limits<double>::quiet_NaN();
        const double a = -0.5 * kGravity;
        const double b = vel[2];
        const double c = pos[2] - kZBounce;
        const double disc = b * b - 4.0 * a * c;
        if (disc < 0.0) return std::numeric_limits<double>::quiet_NaN();
        const double root = std::sqrt(disc);
        const double denom = 2.0 * a;
        const double t1 = (-b - root) / denom;
        const double t2 = (-b + root) / denom;
        double t = std::numeric_limits<double>::quiet_NaN();
        if (t1 > 1e-4) t = t1;
        if (t2 > 1e-4) t = std::isfinite(t) ? std::max(t, t2) : t2;
        if (!std::isfinite(t)) return std::numeric_limits<double>::quiet_NaN();
        return pos[0] + vel[0] * t;
    }

    BallGateConfig cfg_;
    std::optional<double> prev_vz_;
    int own_bounces_ = 0;
    bool hit_seen_ = false;
    int live_run_ = 0;
    int dead_run_ = 0;
    bool engaged_ = false;
    BallGateOutput last_;
};

std::optional<std::array<double, 7>> orderedJointVector(
    const std::vector<std::string>& names,
    const std::vector<double>& values,
    const std::optional<std::array<double, 7>>& fallback) {
    if (values.empty()) return std::nullopt;
    if (names.empty() && values.size() >= 7) {
        std::array<double, 7> out{};
        std::copy_n(values.begin(), 7, out.begin());
        return out;
    }

    std::unordered_map<std::string, size_t> index;
    for (size_t i = 0; i < names.size(); ++i) index[names[i]] = i;
    std::array<double, 7> out = fallback.value_or(std::array<double, 7>{});
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
    if (found == 7) return out;
    if (!fallback.has_value() && values.size() >= 7) {
        std::copy_n(values.begin(), 7, out.begin());
        return out;
    }
    if (found > 0 && fallback.has_value()) return out;
    return std::nullopt;
}

}  // namespace

class A1PolicyBridgeCpp : public rclcpp::Node {
public:
    A1PolicyBridgeCpp() : Node("a1_policy_bridge_cpp") {
        policy_path_ = declare_parameter<std::string>("policy_path", kDefaultPolicy);
        if (policy_path_.empty()) policy_path_ = kDefaultPolicy;
        predictor_path_ = declare_parameter<std::string>("predictor_path", "");
        use_predictor_ = declare_parameter<bool>("use_predictor", true);
        control_hz_ = declare_parameter<double>("control_hz", 50.0);
        joint_timeout_s_ = declare_parameter<double>("joint_timeout_s", 2.0);
        ball_timeout_s_ = declare_parameter<double>("ball_timeout_s", 0.20);
        publish_actions_ = declare_parameter<bool>("publish_actions", true);
        publish_position_velocity_ = declare_parameter<bool>("publish_position_velocity", false);
        servo_filter_enabled_ = declare_parameter<bool>("servo_filter_enabled", true);
        qdes_slew_enabled_ = declare_parameter<bool>("qdes_slew_enabled", false);
        policy_enabled_ = declare_parameter<bool>("policy_enabled_on_start", false);
        enable_on_start_ = declare_parameter<bool>("enable_on_start", false);
        disable_on_stale_joint_ = declare_parameter<bool>("disable_on_stale_joint", true);
        hold_when_ball_stale_ = declare_parameter<bool>("hold_when_ball_stale", false);
        diag_every_ = declare_parameter<int>("diag_every", 50);
        action_topic_ = declare_parameter<std::string>("action_topic", "/model_action");
        enable_topic_ = declare_parameter<std::string>("enable_topic", "/model_control/enable");
        policy_enable_topic_ = declare_parameter<std::string>("policy_enable_topic", "/a1_tt/policy_enable");
        joint_state_topic_ = declare_parameter<std::string>("joint_state_topic", "/right_joint_states");
        ball_state_topic_ = declare_parameter<std::string>("ball_state_topic", "/ball/state");
        auto default_q = declare_parameter<std::vector<double>>(
            "default_q", std::vector<double>(kDefaultRightQ.begin(), kDefaultRightQ.end()));
        if (default_q.size() != 7) {
            throw std::runtime_error("default_q must contain 7 values");
        }
        std::copy(default_q.begin(), default_q.end(), default_right_q_.begin());

        auto robot_table_pos = declare_parameter<std::vector<double>>(
            "robot_table_pos", {-1.8, 0.76, 0.0282});
        if (robot_table_pos.size() != 3) {
            throw std::runtime_error("robot_table_pos must contain 3 values");
        }
        for (size_t i = 0; i < 3; ++i) {
            robot_table_pos_[i] = static_cast<float>(robot_table_pos[i]);
        }
        hit_plane_x_ = declare_parameter<double>("hit_plane_x", kHitPlaneX);
        home_y_ = declare_parameter<double>("home_y", kHomeY);
        paddle_y_offset_ = declare_parameter<double>("paddle_y_offset", kPaddleYOffset);

        auto pred_sentinel = declare_parameter<std::vector<double>>(
            "pred_sentinel",
            {static_cast<double>(kPredSentinel[0]),
             static_cast<double>(kPredSentinel[1]),
             static_cast<double>(kPredSentinel[2])});
        if (pred_sentinel.size() != 3) {
            throw std::runtime_error("pred_sentinel must contain 3 values");
        }
        for (size_t i = 0; i < 3; ++i) {
            pred_sentinel_[i] = static_cast<float>(pred_sentinel[i]);
        }

        auto hit_target_y = declare_parameter<std::vector<double>>(
            "hit_target_y_range", {0.0, 0.55});
        auto hit_target_z = declare_parameter<std::vector<double>>(
            "hit_target_z_range", {0.90, 1.25});
        if (hit_target_y.size() != 2 || hit_target_z.size() != 2) {
            throw std::runtime_error("hit target ranges must contain 2 values each");
        }
        if (hit_target_y[0] > hit_target_y[1] || hit_target_z[0] > hit_target_z[1]) {
            throw std::runtime_error("hit target range lower bounds must not exceed upper bounds");
        }
        std::copy(hit_target_y.begin(), hit_target_y.end(), hit_target_y_range_.begin());
        std::copy(hit_target_z.begin(), hit_target_z.end(), hit_target_z_range_.begin());
        zero_action_when_ball_invalid_ = declare_parameter<bool>("zero_action_when_ball_invalid", false);

        const std::vector<double> default_max_delta_per_tick{
            0.020, 0.024, 0.036, 0.032, 0.080, 0.064, 0.160};
        auto max_delta = declare_parameter<std::vector<double>>("max_delta_per_tick", default_max_delta_per_tick);
        if (max_delta.size() != 7) {
            throw std::runtime_error("max_delta_per_tick must contain 7 values");
        }
        std::copy(max_delta.begin(), max_delta.end(), max_delta_per_tick_.begin());
        // Debug: freeze these joint indices (0-based) to their default pose instead of
        // following the policy output. Runtime-settable via `ros2 param set ... zero_joints "[4,5,6]"`.
        declare_parameter<std::vector<int64_t>>("zero_joints", std::vector<int64_t>{});
        auto servo_vel = declare_parameter<std::vector<double>>(
            "servo_velocity_limit",
            std::vector<double>(kIsaacServoVelocityLimit.begin(), kIsaacServoVelocityLimit.end()));
        if (servo_vel.size() != 7) {
            throw std::runtime_error("servo_velocity_limit must contain 7 values");
        }
        std::copy(servo_vel.begin(), servo_vel.end(), servo_velocity_limit_.begin());
        auto servo_tau = declare_parameter<std::vector<double>>(
            "servo_tau_s",
            std::vector<double>(kIsaacServoTauS.begin(), kIsaacServoTauS.end()));
        if (servo_tau.size() != 7) {
            throw std::runtime_error("servo_tau_s must contain 7 values");
        }
        std::copy(servo_tau.begin(), servo_tau.end(), servo_tau_s_.begin());
        for (double t : servo_tau_s_) {
            if (t <= 0.0) {
                throw std::runtime_error("servo_tau_s values must be > 0");
            }
        }

        BallGateConfig gate_cfg;
        gate_cfg.confirm_frames = declare_parameter<int>("gate_confirm_frames", 1);
        gate_cfg.coast_frames = declare_parameter<int>("gate_coast_frames", 1);
        gate_cfg.vx_away = declare_parameter<double>("gate_min_approach_vx", -0.05);
        gate_cfg.hit_plane_x = hit_plane_x_;
        gate_ = std::make_unique<BallValidityGate>(gate_cfg);

        auto ranges = softJointRanges();
        q_min_ = ranges.first;
        q_max_ = ranges.second;
        last_action_.fill(0.0f);
        last_q_des_ = default_right_q_;
        last_ball_pred_ = pred_sentinel_;
        resetServoFilter(std::nullopt);
        resetPolicy(std::nullopt);

        policy_ = std::make_unique<OrtRunner>(policy_path_);
        if (policy_->inputSize() != static_cast<size_t>(kObsSize)) {
            throw std::runtime_error(
                "policy input must be " + std::to_string(kObsSize) +
                ", got " + std::to_string(policy_->inputSize()) +
                " for input " + policy_->inputName());
        }
        if (predictor_path_.empty()) {
            predictor_path_ = (fs::path(policy_path_).parent_path() / "predictor.onnx").string();
        }
        if (use_predictor_ && fs::exists(predictor_path_)) {
            predictor_ = std::make_unique<OrtRunner>(predictor_path_);
        }

        action_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(action_topic_, 10);
        enable_pub_ = create_publisher<std_msgs::msg::Bool>(enable_topic_, 10);
        raw_action_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/sim2real/raw_action", 10);
        q_des_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/sim2real/q_des", 10);
        raw_q_des_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/sim2real/raw_q_des", 10);
        dq_des_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/sim2real/dq_des", 10);
        obs_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/sim2real/obs", 10);
        frame_pub_ = create_publisher<std_msgs::msg::Float64MultiArray>("/sim2real/frame", 10);
        gate_pub_ = create_publisher<std_msgs::msg::String>("/sim2real/gate", 10);

        joint_sub_ = create_subscription<sensor_msgs::msg::JointState>(
            joint_state_topic_, 10, std::bind(&A1PolicyBridgeCpp::jointCb, this, std::placeholders::_1));
        ball_sub_ = create_subscription<std_msgs::msg::Float64MultiArray>(
            ball_state_topic_, 10, std::bind(&A1PolicyBridgeCpp::ballCb, this, std::placeholders::_1));
        policy_enable_sub_ = create_subscription<std_msgs::msg::Bool>(
            policy_enable_topic_, 10, std::bind(&A1PolicyBridgeCpp::policyEnableCb, this, std::placeholders::_1));

        const double period = 1.0 / std::max(control_hz_, 1e-6);
        timer_ = create_wall_timer(
            std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::duration<double>(period)),
            std::bind(&A1PolicyBridgeCpp::controlTick, this));

        publishEnable(enable_on_start_);
        RCLCPP_INFO(
            get_logger(),
            "a1_policy_bridge_cpp ready: policy=%s predictor=%s joint_topic=%s ball_topic=%s action_topic=%s publish_actions=%s publish_pv=%s servo_filter=%s servo_tau=%s servo_vel=%s qdes_slew=%s max_delta_per_tick=%s",
            policy_path_.c_str(),
            predictor_ ? predictor_path_.c_str() : "(disabled)",
            joint_state_topic_.c_str(),
            ball_state_topic_.c_str(),
            action_topic_.c_str(),
            publish_actions_ ? "true" : "false",
            publishPositionVelocity() ? "true" : "false",
            servo_filter_enabled_ ? "true" : "false",
            vecToString(servo_tau_s_).c_str(),
            vecToString(servo_velocity_limit_).c_str(),
            qdes_slew_enabled_ ? "true" : "false",
            vecToString(max_delta_per_tick_).c_str());
        RCLCPP_INFO(
            get_logger(),
            "policy runtime gate: topic=%s enabled=%s",
            policy_enable_topic_.c_str(),
            policy_enabled_ ? "true" : "false");
        RCLCPP_INFO(
            get_logger(),
            "policy geometry: default_q=%s robot_pos=[%.3f, %.3f, %.4f] hit_plane=%.3f paddle_y_offset=%.3f sentinel=%s target_y=[%.3f, %.3f] target_z=[%.3f, %.3f] invalid_action=%s",
            vecToString(default_right_q_).c_str(),
            robot_table_pos_[0], robot_table_pos_[1], robot_table_pos_[2],
            hit_plane_x_, paddle_y_offset_, vecToString(pred_sentinel_).c_str(),
            hit_target_y_range_[0], hit_target_y_range_[1],
            hit_target_z_range_[0], hit_target_z_range_[1],
            zero_action_when_ball_invalid_ ? "zero" : "policy");
    }

    ~A1PolicyBridgeCpp() override {
        if (rclcpp::ok()) publishEnable(false);
    }

private:
    double nowSec() const {
        return std::chrono::duration<double>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    }

    bool stale(const std::optional<double>& stamp, double timeout_s) const {
        return !stamp.has_value() || nowSec() - stamp.value() > timeout_s;
    }

    void publishEnable(bool enabled) {
        std_msgs::msg::Bool msg;
        msg.data = enabled;
        enable_pub_->publish(msg);
        enabled_sent_ = enabled;
    }

    void jointCb(const sensor_msgs::msg::JointState::SharedPtr msg) {
        auto q = orderedJointVector(
            msg->name,
            msg->position,
            q_.has_value() ? q_ : std::optional<std::array<double, 7>>(default_right_q_));
        if (!q.has_value()) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "joint state did not contain 7 usable right-arm positions");
            return;
        }
        auto dq = orderedJointVector(msg->name, msg->velocity, dq_);
        q_ = q.value();
        dq_ = dq.value_or(std::array<double, 7>{});
        last_joint_time_ = nowSec();
        if (!last_pub_q_.has_value()) {
            last_pub_q_ = q_;
            resetServoFilter(q_);
            resetPolicy(q_);
        }
    }

    void ballCb(const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
        if (msg->data.size() < 3) {
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "ball_state expects data=[x,y,z,vx,vy,vz] or [x,y,z]");
            return;
        }
        const double t = nowSec();
        std::array<double, 3> pos{msg->data[0], msg->data[1], msg->data[2]};
        std::array<double, 3> vel{0.0, 0.0, 0.0};
        if (msg->data.size() >= 6) {
            vel = {msg->data[3], msg->data[4], msg->data[5]};
        } else if (prev_ball_pos_.has_value() && prev_ball_time_.has_value()) {
            const double dt = std::max(t - prev_ball_time_.value(), 1e-6);
            for (size_t i = 0; i < 3; ++i) vel[i] = (pos[i] - prev_ball_pos_.value()[i]) / dt;
        }
        prev_ball_pos_ = pos;
        prev_ball_time_ = t;
        ball_pos_ = pos;
        ball_vel_ = vel;
        last_ball_time_ = t;
    }

    void policyEnableCb(const std_msgs::msg::Bool::SharedPtr msg) {
        if (policy_enabled_ == msg->data) return;
        policy_enabled_ = msg->data;
        if (gate_) {
            gate_->reset();
        }
        if (q_.has_value()) {
            resetPolicy(q_);
            last_pub_q_ = q_;
            resetServoFilter(q_);
        }
        RCLCPP_INFO(
            get_logger(),
            "policy_enabled=%s; reset ball gate",
            policy_enabled_ ? "true" : "false");
    }

    void controlTick() {
        ++tick_;
        if (!q_.has_value()) return;
        if (!policy_enabled_) {
            if (diag_every_ > 0 && tick_ % diag_every_ == 0) {
                RCLCPP_INFO(get_logger(), "tick=%ld policy disabled; suppressing policy action", tick_);
            }
            return;
        }
        if (stale(last_joint_time_, joint_timeout_s_)) {
            if (disable_on_stale_joint_ && enabled_sent_) publishEnable(false);
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "stale joint state; suppressing policy action");
            return;
        }

        std::array<double, 7> q_des{};
        std::array<double, 7> dq_des{};
        std::array<float, 7> raw_action{};
        std::array<float, 3> ball_pred{
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN(),
            std::numeric_limits<float>::quiet_NaN()};
        BallGateOutput gate_out;
        const bool ball_stale = stale(last_ball_time_, ball_timeout_s_);

        try {
            if (ball_stale) {
                if (gate_) {
                    gate_->reset();
                }
                gate_out = gate_->update(nan3(), nan3());
                if (hold_when_ball_stale_) {
                    resetPolicy(q_);
                    q_des = q_.value();
                    raw_action.fill(0.0f);
                } else {
                    auto step = policyStep(q_.value(), dq_, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, false);
                    q_des = step.q_des;
                    raw_action = step.raw_action;
                    ball_pred = step.ball_pred;
                }
            } else {
                gate_out = gate_->update(ball_pos_.value(), ball_vel_.value());
                auto step = policyStep(q_.value(), dq_, ball_pos_.value(), ball_vel_.value(), gate_out.engaged);
                q_des = step.q_des;
                raw_action = step.raw_action;
                ball_pred = step.ball_pred;
            }
        } catch (const std::exception& e) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 1000, "policy step failed: %s", e.what());
            return;
        }

        const auto raw_q_des = clampJointLimits(q_des);
        q_des = commandFromRawQDes(raw_q_des, q_.value(), dq_des);

        if (publish_actions_ && policy_enabled_) {
            std_msgs::msg::Float64MultiArray msg;
            if (publishPositionVelocity()) {
                msg.data.resize(14);
                for (size_t i = 0; i < 7; ++i) {
                    msg.data[i] = q_des[i];
                    msg.data[i + 7] = dq_des[i];
                }
            } else {
                msg.data.assign(q_des.begin(), q_des.end());
            }
            action_pub_->publish(msg);
        }

        publishDiag(raw_action, raw_q_des, q_des, dq_des, gate_out, ball_stale, ball_pred);
    }

    struct PolicyStep {
        std::array<float, 7> raw_action{};
        std::array<double, 7> q_des{};
        std::array<float, 3> ball_pred{};
    };

    PolicyStep policyStep(
        const std::array<double, 7>& q,
        const std::array<double, 7>& dq,
        const std::array<double, 3>& ball_pos,
        const std::array<double, 3>& ball_vel,
        bool valid_ball) {
        const auto obs = observe(q, dq, ball_pos, ball_vel, valid_ball);
        publishPolicyInputDiag(obs, last_frame_);
        const auto raw_vec = policy_->runSingle(obs);
        if (raw_vec.size() < 7) {
            throw std::runtime_error("policy output has fewer than 7 values");
        }
        PolicyStep out;
        for (size_t i = 0; i < 7; ++i) {
            out.raw_action[i] = (!valid_ball && zero_action_when_ball_invalid_) ? 0.0f : raw_vec[i];
        }
        out.q_des = actionToQDes(out.raw_action);
        out.ball_pred = last_ball_pred_;
        return out;
    }

    std::vector<float> observe(
        const std::array<double, 7>& q,
        const std::array<double, 7>& dq,
        const std::array<double, 3>& ball_pos,
        const std::array<double, 3>& ball_vel,
        bool valid_ball) {
        auto frame = computeFrame(q, dq, ball_pos, ball_vel, valid_ball);
        history_.push_back(frame);
        while (history_.size() > kHistory) history_.pop_front();
        std::vector<float> obs;
        obs.reserve(kObsSize);
        for (const auto& f : history_) obs.insert(obs.end(), f.begin(), f.end());
        for (auto& v : obs) v = clampFloat(v, -kClipObs, kClipObs);
        return obs;
    }

    std::array<float, kFrameSize> computeFrame(
        const std::array<double, 7>& q,
        const std::array<double, 7>& dq,
        const std::array<double, 3>& ball_pos,
        const std::array<double, 3>& ball_vel,
        bool valid_ball) {
        std::array<float, kFrameSize> frame{};
        size_t o = 0;
        frame[o++] = 0.0f;
        frame[o++] = 0.0f;
        frame[o++] = 0.0f;
        frame[o++] = 0.0f;
        frame[o++] = 0.0f;
        frame[o++] = -1.0f;
        for (size_t i = 0; i < 7; ++i) frame[o++] = static_cast<float>(q[i] - default_right_q_[i]);
        for (double v : dq) frame[o++] = static_cast<float>(v);
        for (float v : last_action_) frame[o++] = v;

        const auto ball_pred = predictBall(ball_pos, ball_vel, valid_ball);
        last_ball_pred_ = ball_pred;
        const std::array<float, 3> ball_obs = valid_ball
            ? std::array<float, 3>{static_cast<float>(ball_pos[0]), static_cast<float>(ball_pos[1]), static_cast<float>(ball_pos[2])}
            : pred_sentinel_;
        for (float v : ball_obs) frame[o++] = v;
        for (float v : robot_table_pos_) frame[o++] = v;
        for (float v : ball_pred) frame[o++] = v;
        frame[o++] = static_cast<float>((ball_pred[0] - 0.1f) - robot_table_pos_[0]);
        frame[o++] = static_cast<float>(
            (ball_pred[1] - static_cast<float>(paddle_y_offset_)) - robot_table_pos_[1]);
        frame[o++] = 0.0f;
        if (o != static_cast<size_t>(kFrameSize)) {
            throw std::runtime_error("bad frame size");
        }
        last_frame_ = frame;
        return frame;
    }

    std::array<float, 3> predictBall(
        const std::array<double, 3>& ball_pos,
        const std::array<double, 3>& ball_vel,
        bool valid_ball) {
        if (predictor_ && valid_ball && finite3(ball_pos)) {
            auto pred = runPredictor(ball_pos);
            if (plausiblePrediction(pred)) {
                return projectPrediction(pred);
            }
        } else if (!valid_ball) {
            predictor_history_.clear();
        }
        return analyticPrediction(ball_pos, ball_vel, valid_ball);
    }

    std::array<float, 3> runPredictor(const std::array<double, 3>& ball_pos) {
        const std::array<float, 3> p{
            static_cast<float>(ball_pos[0]),
            static_cast<float>(ball_pos[1]),
            static_cast<float>(ball_pos[2])};
        predictor_history_.push_back(p);
        while (predictor_history_.size() > kHistory) predictor_history_.pop_front();
        std::vector<float> input;
        input.reserve(3 * kHistory);
        const auto first = predictor_history_.empty() ? std::array<float, 3>{0.0f, 0.0f, 0.0f} : predictor_history_.front();
        for (size_t i = predictor_history_.size(); i < kHistory; ++i) {
            input.insert(input.end(), first.begin(), first.end());
        }
        for (const auto& row : predictor_history_) {
            input.insert(input.end(), row.begin(), row.end());
        }
        const auto out = predictor_->runSingle(input);
        if (out.size() < 3) return pred_sentinel_;
        return {out[0], out[1], out[2]};
    }

    bool plausiblePrediction(const std::array<float, 3>& pred) const {
        const float y_center = static_cast<float>(home_y_ + paddle_y_offset_);
        return finite3f(pred)
            && pred[0] > static_cast<float>(hit_plane_x_ - 0.50)
            && pred[0] < static_cast<float>(hit_plane_x_ + 0.30)
            && std::abs(pred[1] - y_center) < 0.45f
            && pred[2] > 0.85f
            && pred[2] < 1.55f;
    }

    std::array<float, 3> projectPrediction(std::array<float, 3> pred) const {
        pred[0] = static_cast<float>(hit_plane_x_);
        pred[1] = clampFloat(
            pred[1],
            static_cast<float>(hit_target_y_range_[0]),
            static_cast<float>(hit_target_y_range_[1]));
        pred[2] = clampFloat(
            pred[2],
            static_cast<float>(hit_target_z_range_[0]),
            static_cast<float>(hit_target_z_range_[1]));
        return pred;
    }

    std::array<float, 3> analyticPrediction(
        const std::array<double, 3>& ball_pos,
        const std::array<double, 3>& ball_vel,
        bool valid_ball) const {
        if (!valid_ball || !finite3(ball_pos) || ball_vel[0] >= -0.05) return pred_sentinel_;
        const double t = (hit_plane_x_ - ball_pos[0]) / ball_vel[0];
        if (t <= 0.0 || t > 2.0) return pred_sentinel_;
        std::array<double, 3> pred{
            ball_pos[0] + ball_vel[0] * t,
            ball_pos[1] + ball_vel[1] * t,
            ball_pos[2] + ball_vel[2] * t - 0.5 * kGravity * t * t};
        if (pred[2] < 0.2 || pred[2] > 2.0) return pred_sentinel_;
        return projectPrediction(
            {static_cast<float>(pred[0]), static_cast<float>(pred[1]), static_cast<float>(pred[2])});
    }

    std::array<double, 7> actionToQDes(const std::array<float, 7>& raw_action) {
        last_action_ = raw_action;
        std::vector<int64_t> zero_joints;
        get_parameter("zero_joints", zero_joints);
        std::array<bool, 7> zmask{};
        for (int64_t j : zero_joints) {
            if (j >= 0 && j < 7) zmask[static_cast<size_t>(j)] = true;
        }
        std::array<double, 7> q_des{};
        for (size_t i = 0; i < 7; ++i) {
            if (zmask[i]) {
                q_des[i] = default_right_q_[i];  // freeze to default, ignore policy
                continue;
            }
            const double a = clampDouble(raw_action[i], -kClipActions, kClipActions);
            q_des[i] = clampDouble(a * kActionScale + default_right_q_[i], q_min_[i], q_max_[i]);
        }
        last_q_des_ = q_des;
        return q_des;
    }

    std::array<double, 7> clampJointLimits(const std::array<double, 7>& q) const {
        std::array<double, 7> out{};
        for (size_t i = 0; i < 7; ++i) {
            out[i] = clampDouble(q[i], q_min_[i], q_max_[i]);
        }
        return out;
    }

    void resetPolicy(const std::optional<std::array<double, 7>>& q) {
        predictor_history_.clear();
        last_action_.fill(0.0f);
        if (q.has_value()) last_q_des_ = q.value();
        history_.clear();
        std::array<double, 7> q0 = q.value_or(default_right_q_);
        std::array<double, 7> dq0{};
        std::array<double, 3> zero3{0.0, 0.0, 0.0};
        const auto frame = computeFrame(q0, dq0, zero3, zero3, false);
        for (int i = 0; i < kHistory; ++i) history_.push_back(frame);
    }

    std::array<double, 7> clampDelta(const std::array<double, 7>& q_des) const {
        if (!last_pub_q_.has_value()) return q_des;
        const bool disabled = std::all_of(
            max_delta_per_tick_.begin(), max_delta_per_tick_.end(),
            [](double v) { return v <= 0.0; });
        if (disabled) return q_des;
        std::array<double, 7> out{};
        for (size_t i = 0; i < 7; ++i) {
            const double d = q_des[i] - last_pub_q_.value()[i];
            out[i] = last_pub_q_.value()[i] + clampDouble(d, -max_delta_per_tick_[i], max_delta_per_tick_[i]);
        }
        return out;
    }

    void resetServoFilter(const std::optional<std::array<double, 7>>& q) {
        servo_q_cmd_ = clampJointLimits(q.value_or(default_right_q_));
        servo_dq_cmd_.fill(0.0);
        servo_filter_initialized_ = q.has_value();
        last_pub_q_ = servo_q_cmd_;
    }

    std::array<double, 7> commandFromRawQDes(
        const std::array<double, 7>& raw_q_des,
        const std::array<double, 7>& measured_q,
        std::array<double, 7>& dq_des) {
        if (!servo_filter_enabled_) {
            std::array<double, 7> cmd = qdes_slew_enabled_ ? clampDelta(raw_q_des) : raw_q_des;
            const auto prev = last_pub_q_.value_or(measured_q);
            for (size_t i = 0; i < 7; ++i) {
                dq_des[i] = (cmd[i] - prev[i]) * control_hz_;
            }
            last_pub_q_ = cmd;
            return cmd;
        }

        const double dt = 1.0 / std::max(control_hz_, 1e-6);
        if (!servo_filter_initialized_) {
            servo_q_cmd_ = clampJointLimits(measured_q);
            servo_dq_cmd_.fill(0.0);
            servo_filter_initialized_ = true;
        }
        const auto prev = servo_q_cmd_;
        for (size_t i = 0; i < 7; ++i) {
            const double desired_dq = (raw_q_des[i] - servo_q_cmd_[i]) / servo_tau_s_[i];
            servo_dq_cmd_[i] = clampDouble(
                desired_dq,
                -std::abs(servo_velocity_limit_[i]),
                std::abs(servo_velocity_limit_[i]));
            servo_q_cmd_[i] += servo_dq_cmd_[i] * dt;
        }
        servo_q_cmd_ = clampJointLimits(servo_q_cmd_);
        for (size_t i = 0; i < 7; ++i) {
            servo_dq_cmd_[i] = (servo_q_cmd_[i] - prev[i]) / dt;
            dq_des[i] = servo_dq_cmd_[i];
        }
        last_pub_q_ = servo_q_cmd_;
        return servo_q_cmd_;
    }

    bool publishPositionVelocity() const {
        return publish_position_velocity_ || servo_filter_enabled_;
    }

    void publishDiag(
        const std::array<float, 7>& raw_action,
        const std::array<double, 7>& raw_q_des,
        const std::array<double, 7>& q_des,
        const std::array<double, 7>& dq_des,
        const BallGateOutput& gate_out,
        bool ball_stale,
        const std::array<float, 3>& ball_pred) {
        std_msgs::msg::Float64MultiArray raw_msg;
        raw_msg.data.reserve(7);
        for (float v : raw_action) raw_msg.data.push_back(static_cast<double>(v));
        raw_action_pub_->publish(raw_msg);

        std_msgs::msg::Float64MultiArray q_msg;
        q_msg.data.assign(q_des.begin(), q_des.end());
        q_des_pub_->publish(q_msg);

        std_msgs::msg::Float64MultiArray raw_q_msg;
        raw_q_msg.data.assign(raw_q_des.begin(), raw_q_des.end());
        raw_q_des_pub_->publish(raw_q_msg);

        std_msgs::msg::Float64MultiArray dq_msg;
        dq_msg.data.assign(dq_des.begin(), dq_des.end());
        dq_des_pub_->publish(dq_msg);

        std_msgs::msg::String gate_msg;
        std::ostringstream os;
        os.setf(std::ios::fixed);
        os.precision(3);
        os << "engaged=" << (gate_out.engaged ? 1 : 0)
           << " live=" << (gate_out.live ? 1 : 0)
           << " reason=" << gate_out.reason
           << " first_bounce_x=" << gate_out.first_bounce_x
           << " own_bounces=" << gate_out.own_bounces
           << " ball_stale=" << (ball_stale ? 1 : 0)
           << " pred=" << vecToString(ball_pred);
        gate_msg.data = os.str();
        gate_pub_->publish(gate_msg);

        if (diag_every_ > 0 && tick_ % diag_every_ == 0) {
            float max_raw = 0.0f;
            for (float v : raw_action) max_raw = std::max(max_raw, std::abs(v));
            RCLCPP_INFO(
                get_logger(),
                "tick=%ld gate=%d/%s raw_q_des=%s q_des=%s dq_des=%s raw_max=%.2f pred=%s",
                tick_,
                gate_out.engaged ? 1 : 0,
                gate_out.reason.c_str(),
                vecToString(raw_q_des).c_str(),
                vecToString(q_des).c_str(),
                vecToString(dq_des).c_str(),
                max_raw,
                vecToString(ball_pred).c_str());
        }
    }

    void publishPolicyInputDiag(
        const std::vector<float>& obs,
        const std::array<float, kFrameSize>& frame) {
        std_msgs::msg::Float64MultiArray obs_msg;
        obs_msg.data.reserve(obs.size());
        for (float v : obs) obs_msg.data.push_back(static_cast<double>(v));
        obs_pub_->publish(obs_msg);

        std_msgs::msg::Float64MultiArray frame_msg;
        frame_msg.data.reserve(frame.size());
        for (float v : frame) frame_msg.data.push_back(static_cast<double>(v));
        frame_pub_->publish(frame_msg);
    }

    static std::array<double, 3> nan3() {
        const double n = std::numeric_limits<double>::quiet_NaN();
        return {n, n, n};
    }

    std::string policy_path_;
    std::string predictor_path_;
    bool use_predictor_ = true;
    double control_hz_ = 50.0;
    double joint_timeout_s_ = 2.0;
    double ball_timeout_s_ = 0.20;
    bool publish_actions_ = true;
    bool publish_position_velocity_ = false;
    bool servo_filter_enabled_ = true;
    std::array<double, 7> servo_tau_s_ = kIsaacServoTauS;
    bool qdes_slew_enabled_ = false;
    bool policy_enabled_ = false;
    bool enable_on_start_ = false;
    bool disable_on_stale_joint_ = true;
    bool hold_when_ball_stale_ = false;
    int diag_every_ = 50;
    std::string action_topic_;
    std::string enable_topic_;
    std::string policy_enable_topic_;
    std::string joint_state_topic_;
    std::string ball_state_topic_;
    std::array<double, 7> default_right_q_ = kDefaultRightQ;
    std::array<float, 3> robot_table_pos_ = kRobotTablePos;
    double hit_plane_x_ = kHitPlaneX;
    double home_y_ = kHomeY;
    double paddle_y_offset_ = kPaddleYOffset;
    std::array<float, 3> pred_sentinel_ = kPredSentinel;
    std::array<double, 2> hit_target_y_range_{0.0, 0.55};
    std::array<double, 2> hit_target_z_range_{0.90, 1.25};
    bool zero_action_when_ball_invalid_ = false;

    std::unique_ptr<OrtRunner> policy_;
    std::unique_ptr<OrtRunner> predictor_;
    std::unique_ptr<BallValidityGate> gate_;
    std::deque<std::array<float, kFrameSize>> history_;
    std::deque<std::array<float, 3>> predictor_history_;
    std::array<double, 7> q_min_{};
    std::array<double, 7> q_max_{};
    std::array<float, 7> last_action_{};
    std::array<double, 7> last_q_des_{};
    std::array<float, kFrameSize> last_frame_{};
    std::array<float, 3> last_ball_pred_ = kPredSentinel;

    std::optional<std::array<double, 7>> q_;
    std::array<double, 7> dq_{};
    std::optional<double> last_joint_time_;
    std::optional<std::array<double, 3>> ball_pos_;
    std::optional<std::array<double, 3>> ball_vel_;
    std::optional<double> last_ball_time_;
    std::optional<std::array<double, 3>> prev_ball_pos_;
    std::optional<double> prev_ball_time_;
    std::optional<std::array<double, 7>> last_pub_q_;
    std::array<double, 7> max_delta_per_tick_{};
    std::array<double, 7> servo_velocity_limit_ = kIsaacServoVelocityLimit;
    std::array<double, 7> servo_q_cmd_ = kDefaultRightQ;
    std::array<double, 7> servo_dq_cmd_{};
    bool servo_filter_initialized_ = false;
    long tick_ = 0;
    bool enabled_sent_ = false;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr action_pub_;
    rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr enable_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr raw_action_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr q_des_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr raw_q_des_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr dq_des_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr obs_pub_;
    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr frame_pub_;
    rclcpp::Publisher<std_msgs::msg::String>::SharedPtr gate_pub_;
    rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_sub_;
    rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr ball_sub_;
    rclcpp::Subscription<std_msgs::msg::Bool>::SharedPtr policy_enable_sub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    try {
        auto node = std::make_shared<A1PolicyBridgeCpp>();
        rclcpp::spin(node);
    } catch (const std::exception& e) {
        std::cerr << "a1_policy_bridge_cpp failed: " << e.what() << std::endl;
        rclcpp::shutdown();
        return 1;
    }
    rclcpp::shutdown();
    return 0;
}
