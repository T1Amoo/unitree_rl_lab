#include <algorithm>
#include <array>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <geometry_msgs/msg/pose_stamped.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>

namespace {

std::array<double, 3> vec3Param(
    rclcpp::Node& node, const std::string& name, const std::array<double, 3>& fallback) {
    auto values = node.declare_parameter<std::vector<double>>(
        name, std::vector<double>(fallback.begin(), fallback.end()));
    if (values.size() != 3) {
        RCLCPP_WARN(
            node.get_logger(),
            "%s must have 3 values; using fallback [%.3f, %.3f, %.3f]",
            name.c_str(), fallback[0], fallback[1], fallback[2]);
        return fallback;
    }
    return {values[0], values[1], values[2]};
}

std::array<double, 4> quatParam(
    rclcpp::Node& node, const std::string& name, const std::array<double, 4>& fallback) {
    auto values = node.declare_parameter<std::vector<double>>(
        name, std::vector<double>(fallback.begin(), fallback.end()));
    if (values.size() != 4) {
        RCLCPP_WARN(
            node.get_logger(),
            "%s must have 4 values; using fallback [%.3f, %.3f, %.3f, %.3f]",
            name.c_str(), fallback[0], fallback[1], fallback[2], fallback[3]);
        return fallback;
    }
    return {values[0], values[1], values[2], values[3]};
}

std::string vecToString(const std::array<double, 3>& v) {
    std::ostringstream os;
    os.setf(std::ios::fixed);
    os.precision(3);
    os << "[" << v[0] << ", " << v[1] << ", " << v[2] << "]";
    return os.str();
}

}  // namespace

class A1VrpnBallStateBridge : public rclcpp::Node {
public:
    A1VrpnBallStateBridge() : Node("a1_vrpn_ball_state_bridge") {
        input_topic_ = declare_parameter<std::string>("input_topic", "/vrpn_mocap/U_Tracker0/pose");
        output_topic_ = declare_parameter<std::string>("output_topic", "/ball/state");
        origin_ = vec3Param(*this, "origin_in_training_world", {0.0, 0.0, 0.76});
        const auto quat = quatParam(*this, "rotation_wxyz_to_training", {1.0, 0.0, 0.0, 0.0});
        velocity_lpf_alpha_ = declare_parameter<double>("velocity_lpf_alpha", 0.35);
        min_dt_s_ = declare_parameter<double>("min_dt_s", 0.001);
        max_dt_s_ = declare_parameter<double>("max_dt_s", 0.10);
        max_speed_mps_ = declare_parameter<double>("max_speed_mps", 20.0);
        use_header_stamp_ = declare_parameter<bool>("use_header_stamp", true);
        diag_every_ = declare_parameter<int>("diag_every", 50);

        velocity_lpf_alpha_ = clamp(velocity_lpf_alpha_, 0.0, 1.0);
        min_dt_s_ = std::max(1e-6, min_dt_s_);
        max_dt_s_ = std::max(min_dt_s_, max_dt_s_);
        max_speed_mps_ = std::max(0.1, max_speed_mps_);
        diag_every_ = std::max(0, diag_every_);
        setRotation(quat);

        pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(output_topic_, 10);
        auto qos = rclcpp::SensorDataQoS().keep_last(50);
        sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            input_topic_, qos,
            [this](geometry_msgs::msg::PoseStamped::SharedPtr msg) { poseCb(*msg); });

        RCLCPP_INFO(
            get_logger(),
            "ready: %s geometry_msgs/PoseStamped -> %s Float64MultiArray [x,y,z,vx,vy,vz], origin=%s",
            input_topic_.c_str(), output_topic_.c_str(), vecToString(origin_).c_str());
    }

private:
    static double clamp(double v, double lo, double hi) {
        return std::max(lo, std::min(hi, v));
    }

    void setRotation(std::array<double, 4> q) {
        double& w = q[0];
        double& x = q[1];
        double& y = q[2];
        double& z = q[3];
        const double n = std::sqrt(w * w + x * x + y * y + z * z);
        if (n > 1e-12) {
            w /= n;
            x /= n;
            y /= n;
            z /= n;
        } else {
            w = 1.0;
            x = y = z = 0.0;
        }

        rot_[0][0] = 1.0 - 2.0 * (y * y + z * z);
        rot_[0][1] = 2.0 * (x * y - w * z);
        rot_[0][2] = 2.0 * (x * z + w * y);
        rot_[1][0] = 2.0 * (x * y + w * z);
        rot_[1][1] = 1.0 - 2.0 * (x * x + z * z);
        rot_[1][2] = 2.0 * (y * z - w * x);
        rot_[2][0] = 2.0 * (x * z - w * y);
        rot_[2][1] = 2.0 * (y * z + w * x);
        rot_[2][2] = 1.0 - 2.0 * (x * x + y * y);
    }

    std::array<double, 3> toTraining(double x, double y, double z) const {
        return {
            rot_[0][0] * x + rot_[0][1] * y + rot_[0][2] * z + origin_[0],
            rot_[1][0] * x + rot_[1][1] * y + rot_[1][2] * z + origin_[1],
            rot_[2][0] * x + rot_[2][1] * y + rot_[2][2] * z + origin_[2],
        };
    }

    double sampleTimeSec(const geometry_msgs::msg::PoseStamped& msg) const {
        if (use_header_stamp_ && (msg.header.stamp.sec != 0 || msg.header.stamp.nanosec != 0)) {
            return rclcpp::Time(msg.header.stamp).seconds();
        }
        return now().seconds();
    }

    std::array<double, 3> velocityFrom(const std::array<double, 3>& pos, double t) {
        if (!last_pos_valid_) {
            last_pos_ = pos;
            last_t_ = t;
            last_pos_valid_ = true;
            return {0.0, 0.0, 0.0};
        }

        const double dt = t - last_t_;
        if (!std::isfinite(dt) || dt < min_dt_s_ || dt > max_dt_s_) {
            last_pos_ = pos;
            last_t_ = t;
            return filt_vel_;
        }

        std::array<double, 3> raw{
            (pos[0] - last_pos_[0]) / dt,
            (pos[1] - last_pos_[1]) / dt,
            (pos[2] - last_pos_[2]) / dt,
        };
        const double speed = std::sqrt(raw[0] * raw[0] + raw[1] * raw[1] + raw[2] * raw[2]);
        if (speed > max_speed_mps_) {
            const double s = max_speed_mps_ / speed;
            raw[0] *= s;
            raw[1] *= s;
            raw[2] *= s;
        }

        for (size_t i = 0; i < 3; ++i) {
            filt_vel_[i] = velocity_lpf_alpha_ * raw[i] + (1.0 - velocity_lpf_alpha_) * filt_vel_[i];
        }
        last_pos_ = pos;
        last_t_ = t;
        return filt_vel_;
    }

    void poseCb(const geometry_msgs::msg::PoseStamped& msg) {
        const auto pos = toTraining(
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z);
        const auto vel = velocityFrom(pos, sampleTimeSec(msg));

        std_msgs::msg::Float64MultiArray out;
        out.data = {pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]};
        pub_->publish(out);

        ++n_;
        if (diag_every_ > 0 && n_ % static_cast<size_t>(diag_every_) == 0) {
            RCLCPP_INFO(
                get_logger(),
                "ball_state pos=%s vel=%s",
                vecToString(pos).c_str(), vecToString(vel).c_str());
        }
    }

    std::string input_topic_;
    std::string output_topic_;
    std::array<double, 3> origin_{0.0, 0.0, 0.76};
    double rot_[3][3] = {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}};
    double velocity_lpf_alpha_ = 0.35;
    double min_dt_s_ = 0.001;
    double max_dt_s_ = 0.10;
    double max_speed_mps_ = 20.0;
    bool use_header_stamp_ = true;
    int diag_every_ = 50;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_;

    bool last_pos_valid_ = false;
    std::array<double, 3> last_pos_{0.0, 0.0, 0.0};
    std::array<double, 3> filt_vel_{0.0, 0.0, 0.0};
    double last_t_ = 0.0;
    size_t n_ = 0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<A1VrpnBallStateBridge>());
    rclcpp::shutdown();
    return 0;
}
