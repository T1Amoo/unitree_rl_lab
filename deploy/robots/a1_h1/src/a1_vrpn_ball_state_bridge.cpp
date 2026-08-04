#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
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
        temporal_filter_enabled_ = declare_parameter<bool>("temporal_filter_enabled", true);
        filter_alpha_ = declare_parameter<double>("filter_alpha", 0.65);
        filter_beta_ = declare_parameter<double>("filter_beta", 0.10);
        max_innovation_m_ = declare_parameter<double>("max_innovation_m", 0.12);
        reacquire_innovation_m_ = declare_parameter<double>("reacquire_innovation_m", 0.06);
        reacquire_max_speed_mps_ = declare_parameter<double>("reacquire_max_speed_mps", 8.0);
        acquire_frames_ = declare_parameter<int>("acquire_frames", 3);
        reacquire_frames_ = declare_parameter<int>("reacquire_frames", 5);
        reset_gap_s_ = declare_parameter<double>("reset_gap_s", 0.25);
        min_source_age_s_ = declare_parameter<double>("min_source_age_s", 0.01);
        max_source_age_s_ = declare_parameter<double>("max_source_age_s", 0.09);
        max_extrapolation_s_ = declare_parameter<double>("max_extrapolation_s", 0.16);
        gravity_mps2_ = declare_parameter<double>("gravity_mps2", -9.81);
        table_bounce_enabled_ = declare_parameter<bool>("table_bounce_enabled", true);
        table_ball_center_z_ = declare_parameter<double>("table_ball_center_z", 0.78);
        table_restitution_ = declare_parameter<double>("table_restitution", 0.95);
        use_header_stamp_ = declare_parameter<bool>("use_header_stamp", true);
        diag_every_ = declare_parameter<int>("diag_every", 50);

        velocity_lpf_alpha_ = clamp(velocity_lpf_alpha_, 0.0, 1.0);
        min_dt_s_ = std::max(1e-6, min_dt_s_);
        max_dt_s_ = std::max(min_dt_s_, max_dt_s_);
        max_speed_mps_ = std::max(0.1, max_speed_mps_);
        filter_alpha_ = clamp(filter_alpha_, 0.0, 1.0);
        filter_beta_ = clamp(filter_beta_, 0.0, 1.0);
        max_innovation_m_ = std::max(0.001, max_innovation_m_);
        reacquire_innovation_m_ = std::max(0.001, reacquire_innovation_m_);
        reacquire_max_speed_mps_ = std::max(0.1, reacquire_max_speed_mps_);
        acquire_frames_ = std::max(2, acquire_frames_);
        reacquire_frames_ = std::max(2, reacquire_frames_);
        reset_gap_s_ = std::max(max_dt_s_, reset_gap_s_);
        min_source_age_s_ = std::max(-0.02, min_source_age_s_);
        max_source_age_s_ = std::max(min_source_age_s_, max_source_age_s_);
        max_extrapolation_s_ = std::max(0.0, max_extrapolation_s_);
        table_restitution_ = clamp(table_restitution_, 0.0, 1.2);
        diag_every_ = std::max(0, diag_every_);
        setRotation(quat);

        pub_ = create_publisher<std_msgs::msg::Float64MultiArray>(output_topic_, 10);
        auto qos = rclcpp::SensorDataQoS().keep_last(50);
        sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
            input_topic_, qos,
            [this](geometry_msgs::msg::PoseStamped::SharedPtr msg) { poseCb(*msg); });

        RCLCPP_INFO(
            get_logger(),
            "ready: %s -> %s [x,y,z,vx,vy,vz], origin=%s temporal_filter=%s "
            "alpha=%.2f beta=%.2f innovation=%.3fm acquire=%d reacquire=%d "
            "source_age=[%.3f,%.3f]s extrapolate<=%.3fs table_bounce=%s@z=%.3f/e=%.2f",
            input_topic_.c_str(), output_topic_.c_str(), vecToString(origin_).c_str(),
            temporal_filter_enabled_ ? "true" : "false", filter_alpha_, filter_beta_,
            max_innovation_m_, acquire_frames_, reacquire_frames_,
            min_source_age_s_, max_source_age_s_, max_extrapolation_s_,
            table_bounce_enabled_ ? "true" : "false", table_ball_center_z_, table_restitution_);
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

    static double norm3(const std::array<double, 3>& v) {
        return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
    }

    std::pair<std::array<double, 3>, std::array<double, 3>> propagate(
        const std::array<double, 3>& pos,
        const std::array<double, 3>& vel,
        double dt) const {
        dt = std::max(0.0, dt);
        auto out_pos = pos;
        auto out_vel = vel;
        double pre_bounce_dt = dt;
        double post_bounce_dt = 0.0;

        // A camera interval can straddle the table impact. Without this one-
        // bounce model, the predictor falls through the table and the first
        // genuine post-bounce sample looks like a large stereo outlier.
        if (table_bounce_enabled_ && dt > 0.0 && pos[2] >= table_ball_center_z_ - 0.02) {
            const double a = 0.5 * gravity_mps2_;
            const double b = vel[2];
            const double c = pos[2] - table_ball_center_z_;
            const double disc = b * b - 4.0 * a * c;
            if (std::abs(a) > 1e-12 && disc >= 0.0) {
                const double root = std::sqrt(disc);
                const double t1 = (-b - root) / (2.0 * a);
                const double t2 = (-b + root) / (2.0 * a);
                double impact_t = std::numeric_limits<double>::infinity();
                if (t1 >= -1e-6) impact_t = std::min(impact_t, std::max(0.0, t1));
                if (t2 >= -1e-6) impact_t = std::min(impact_t, std::max(0.0, t2));
                if (impact_t <= dt && vel[2] + gravity_mps2_ * impact_t < 0.0) {
                    pre_bounce_dt = impact_t;
                    post_bounce_dt = dt - impact_t;
                }
            }
        }

        out_pos[0] += vel[0] * dt;
        out_pos[1] += vel[1] * dt;
        out_pos[2] += vel[2] * pre_bounce_dt
            + 0.5 * gravity_mps2_ * pre_bounce_dt * pre_bounce_dt;
        out_vel[2] += gravity_mps2_ * pre_bounce_dt;
        if (post_bounce_dt > 0.0) {
            out_pos[2] = table_ball_center_z_;
            out_vel[2] = -table_restitution_ * out_vel[2];
            out_pos[2] += out_vel[2] * post_bounce_dt
                + 0.5 * gravity_mps2_ * post_bounce_dt * post_bounce_dt;
            out_vel[2] += gravity_mps2_ * post_bounce_dt;
        }
        return {out_pos, out_vel};
    }

    void clampSpeed(std::array<double, 3>& vel) const {
        const double speed = norm3(vel);
        if (speed <= max_speed_mps_ || speed <= 1e-9) return;
        const double scale = max_speed_mps_ / speed;
        for (double& value : vel) value *= scale;
    }

    void initializeFilter(
        const std::array<double, 3>& pos,
        double t,
        const std::optional<std::array<double, 3>>& velocity = std::nullopt) {
        filter_pos_ = pos;
        filter_vel_ = velocity.value_or(std::array<double, 3>{0.0, 0.0, 0.0});
        clampSpeed(filter_vel_);
        filter_t_ = t;
        filter_valid_ = true;
        candidate_valid_ = false;
        candidate_count_ = 0;
        candidate_vel_.fill(0.0);
    }

    void startCandidate(const std::array<double, 3>& pos, double t) {
        candidate_pos_ = pos;
        candidate_t_ = t;
        candidate_count_ = 1;
        candidate_valid_ = true;
        candidate_vel_.fill(0.0);
    }

    std::optional<std::array<double, 3>> updateCandidate(
        const std::array<double, 3>& pos, double t, int required_frames) {
        ++pending_measurements_;
        if (!candidate_valid_ || !std::isfinite(t) || t <= candidate_t_
            || t - candidate_t_ > max_dt_s_) {
            startCandidate(pos, t);
            return std::nullopt;
        }

        const double dt = t - candidate_t_;
        if (candidate_count_ == 1) {
            for (size_t i = 0; i < 3; ++i) {
                candidate_vel_[i] = (pos[i] - candidate_pos_[i]) / dt;
            }
            if (norm3(candidate_vel_) > reacquire_max_speed_mps_) {
                startCandidate(pos, t);
                return std::nullopt;
            }
        } else {
            const auto predicted = propagate(candidate_pos_, candidate_vel_, dt);
            std::array<double, 3> innovation{};
            for (size_t i = 0; i < 3; ++i) innovation[i] = pos[i] - predicted.first[i];
            if (norm3(innovation) > reacquire_innovation_m_) {
                startCandidate(pos, t);
                return std::nullopt;
            }
            candidate_vel_ = predicted.second;
            for (size_t i = 0; i < 3; ++i) {
                candidate_vel_[i] += 0.10 * innovation[i] / dt;
            }
            clampSpeed(candidate_vel_);
        }

        candidate_pos_ = pos;
        candidate_t_ = t;
        ++candidate_count_;
        if (candidate_count_ < required_frames) return std::nullopt;
        return candidate_vel_;
    }

    bool acceptMeasurement(const std::array<double, 3>& pos, double t) {
        if (!std::isfinite(t)) return false;
        if (!filter_valid_) {
            const auto velocity = updateCandidate(pos, t, acquire_frames_);
            if (!velocity.has_value()) return false;
            initializeFilter(pos, t, velocity);
            ++accepted_measurements_;
            return true;
        }
        if (t <= filter_t_) {
            ++rejected_measurements_;
            return false;
        }
        if (t - filter_t_ > reset_gap_s_) {
            filter_valid_ = false;
            candidate_valid_ = false;
            const auto velocity = updateCandidate(pos, t, acquire_frames_);
            if (!velocity.has_value()) return false;
            initializeFilter(pos, t, velocity);
            ++accepted_measurements_;
            return true;
        }

        const double dt = t - filter_t_;
        const auto predicted = propagate(filter_pos_, filter_vel_, dt);
        std::array<double, 3> innovation{};
        for (size_t i = 0; i < 3; ++i) innovation[i] = pos[i] - predicted.first[i];
        const double innovation_norm = norm3(innovation);
        last_innovation_m_ = innovation_norm;

        if (innovation_norm <= max_innovation_m_) {
            filter_pos_ = predicted.first;
            filter_vel_ = predicted.second;
            for (size_t i = 0; i < 3; ++i) {
                filter_pos_[i] += filter_alpha_ * innovation[i];
                filter_vel_[i] += filter_beta_ * innovation[i] / std::max(dt, min_dt_s_);
            }
            clampSpeed(filter_vel_);
            filter_t_ = t;
            candidate_valid_ = false;
            candidate_count_ = 0;
            ++accepted_measurements_;
            return true;
        }

        ++rejected_measurements_;
        const auto velocity = updateCandidate(pos, t, reacquire_frames_);
        if (!velocity.has_value()) return false;
        initializeFilter(pos, t, velocity);
        ++accepted_measurements_;
        return true;
    }

    void poseCb(const geometry_msgs::msg::PoseStamped& msg) {
        const auto pos = toTraining(
            msg.pose.position.x, msg.pose.position.y, msg.pose.position.z);
        const double source_t = sampleTimeSec(msg);
        const double receive_t = now().seconds();
        const double source_age = receive_t - source_t;
        if (!std::isfinite(source_age)
            || source_age < min_source_age_s_
            || source_age > max_source_age_s_) {
            ++stale_measurements_;
            RCLCPP_WARN_THROTTLE(
                get_logger(), *get_clock(), 1000,
                "drop camera sample with source_age=%.1fms (allowed=[%.1f, %.1f]ms); "
                "check Jetson/local clock sync and camera freshness",
                1000.0 * source_age,
                1000.0 * min_source_age_s_, 1000.0 * max_source_age_s_);
            return;
        }

        std::array<double, 3> publish_pos = pos;
        std::array<double, 3> publish_vel{};
        if (temporal_filter_enabled_) {
            if (!acceptMeasurement(pos, source_t)) return;
            const double extrapolation = clamp(source_age, 0.0, max_extrapolation_s_);
            const auto current = propagate(filter_pos_, filter_vel_, extrapolation);
            publish_pos = current.first;
            publish_vel = current.second;
            clampSpeed(publish_vel);
        } else {
            publish_vel = velocityFrom(pos, source_t);
        }

        std_msgs::msg::Float64MultiArray out;
        out.data = {
            publish_pos[0], publish_pos[1], publish_pos[2],
            publish_vel[0], publish_vel[1], publish_vel[2]};
        pub_->publish(out);

        ++n_;
        if (diag_every_ > 0 && n_ % static_cast<size_t>(diag_every_) == 0) {
            RCLCPP_INFO(
                get_logger(),
                "ball_state pos=%s vel=%s source_age=%.1fms innovation=%.3fm "
                "accepted=%zu rejected=%zu pending=%zu stale=%zu",
                vecToString(publish_pos).c_str(), vecToString(publish_vel).c_str(),
                1000.0 * source_age, last_innovation_m_, accepted_measurements_,
                rejected_measurements_, pending_measurements_, stale_measurements_);
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
    bool temporal_filter_enabled_ = true;
    double filter_alpha_ = 0.65;
    double filter_beta_ = 0.10;
    double max_innovation_m_ = 0.12;
    double reacquire_innovation_m_ = 0.06;
    double reacquire_max_speed_mps_ = 8.0;
    int acquire_frames_ = 3;
    int reacquire_frames_ = 5;
    double reset_gap_s_ = 0.25;
    double min_source_age_s_ = 0.01;
    double max_source_age_s_ = 0.09;
    double max_extrapolation_s_ = 0.16;
    double gravity_mps2_ = -9.81;
    bool table_bounce_enabled_ = true;
    double table_ball_center_z_ = 0.78;
    double table_restitution_ = 0.95;
    bool use_header_stamp_ = true;
    int diag_every_ = 50;

    rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr pub_;
    rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr sub_;

    bool last_pos_valid_ = false;
    std::array<double, 3> last_pos_{0.0, 0.0, 0.0};
    std::array<double, 3> filt_vel_{0.0, 0.0, 0.0};
    double last_t_ = 0.0;
    size_t n_ = 0;

    bool filter_valid_ = false;
    std::array<double, 3> filter_pos_{0.0, 0.0, 0.0};
    std::array<double, 3> filter_vel_{0.0, 0.0, 0.0};
    double filter_t_ = 0.0;
    bool candidate_valid_ = false;
    std::array<double, 3> candidate_pos_{0.0, 0.0, 0.0};
    std::array<double, 3> candidate_vel_{0.0, 0.0, 0.0};
    double candidate_t_ = 0.0;
    int candidate_count_ = 0;
    double last_innovation_m_ = 0.0;
    size_t accepted_measurements_ = 0;
    size_t rejected_measurements_ = 0;
    size_t pending_measurements_ = 0;
    size_t stale_measurements_ = 0;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<A1VrpnBallStateBridge>());
    rclcpp::shutdown();
    return 0;
}
