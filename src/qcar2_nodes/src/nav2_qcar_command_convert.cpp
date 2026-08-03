#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <functional>
#include <iterator>
#include <memory>
#include <mutex>

#include "geometry_msgs/msg/twist.hpp"
#include "qcar2_interfaces/msg/boolean_leds.hpp"
#include "qcar2_interfaces/msg/motor_commands.hpp"
#include "rclcpp/rclcpp.hpp"


using namespace std::chrono_literals;


class Nav2QCarConverter : public rclcpp::Node
{
public:
    Nav2QCarConverter()
    : Node("nav2_qcar2_command_converter")
    {
        command_timeout_sec_ = this->declare_parameter<double>(
            "command_timeout_sec", 0.25);
        publish_period_ms_ = this->declare_parameter<int64_t>(
            "publish_period_ms", 20);
        steering_scale_ = this->declare_parameter<double>(
            "steering_scale", 1.0);
        steering_offset_ = this->declare_parameter<double>(
            "steering_offset", 0.0);
        min_speed_ = this->declare_parameter<double>("min_speed", -0.60);
        max_speed_ = this->declare_parameter<double>("max_speed", 0.60);
        min_steering_ = this->declare_parameter<double>("min_steering", -0.58);
        max_steering_ = this->declare_parameter<double>("max_steering", 0.58);

        validate_parameters();

        command_publisher_ =
            this->create_publisher<qcar2_interfaces::msg::MotorCommands>(
            "qcar2_motor_speed_cmd", 1);
        led_publisher_ =
            this->create_publisher<qcar2_interfaces::msg::BooleanLeds>(
            "qcar2_led_cmd", 10);

        nav2_subscriber_ = this->create_subscription<geometry_msgs::msg::Twist>(
            "/cmd_vel_nav",
            1,
            std::bind(
                &Nav2QCarConverter::nav2_command_callback,
                this,
                std::placeholders::_1));

        command_timer_ = this->create_wall_timer(
            std::chrono::milliseconds(publish_period_ms_),
            std::bind(&Nav2QCarConverter::command_publish, this));
        led_timer_ = this->create_wall_timer(
            33ms,
            std::bind(&Nav2QCarConverter::led_publish, this));

        last_command_time_ = std::chrono::steady_clock::now();

        RCLCPP_INFO(
            this->get_logger(),
            "QCar command converter ready: timeout=%.3f s, period=%ld ms, "
            "speed=[%.3f, %.3f], steering=[%.3f, %.3f], "
            "steering_scale=%.3f, steering_offset=%.3f",
            command_timeout_sec_,
            static_cast<long>(publish_period_ms_),
            min_speed_,
            max_speed_,
            min_steering_,
            max_steering_,
            steering_scale_,
            steering_offset_);
    }

    ~Nav2QCarConverter() override
    {
        publish_stop_best_effort();
    }

    void publish_stop_best_effort() noexcept
    {
        {
            const std::lock_guard<std::mutex> lock(state_mutex_);
            requested_speed_ = 0.0;
            requested_steering_ = 0.0;
            output_speed_ = 0.0;
            output_steering_ = 0.0;
            has_valid_command_ = false;
        }

        try {
            publish_motor_command(0.0, 0.0);
        } catch (const std::exception & error) {
            RCLCPP_WARN(
                this->get_logger(),
                "Unable to publish shutdown stop command: %s",
                error.what());
        } catch (...) {
            RCLCPP_WARN(
                this->get_logger(),
                "Unable to publish shutdown stop command: unknown error");
        }
    }

private:
    static double clamp(double value, double lower, double upper)
    {
        return std::max(lower, std::min(value, upper));
    }

    void validate_parameters()
    {
        if (!std::isfinite(command_timeout_sec_) || command_timeout_sec_ <= 0.0) {
            RCLCPP_WARN(
                this->get_logger(),
                "Invalid command_timeout_sec; using 0.25 seconds");
            command_timeout_sec_ = 0.25;
        }

        if (publish_period_ms_ <= 0 || publish_period_ms_ > 1000) {
            RCLCPP_WARN(
                this->get_logger(),
                "Invalid publish_period_ms; using 20 milliseconds");
            publish_period_ms_ = 20;
        }

        if (!std::isfinite(steering_scale_)) {
            RCLCPP_WARN(
                this->get_logger(),
                "Invalid steering_scale; using 1.0");
            steering_scale_ = 1.0;
        }

        if (!std::isfinite(steering_offset_)) {
            RCLCPP_WARN(
                this->get_logger(),
                "Invalid steering_offset; using 0.0");
            steering_offset_ = 0.0;
        }

        if (
            !std::isfinite(min_speed_) ||
            !std::isfinite(max_speed_) ||
            min_speed_ > max_speed_)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Invalid speed bounds; using [-0.60, 0.60]");
            min_speed_ = -0.60;
            max_speed_ = 0.60;
        }

        if (
            !std::isfinite(min_steering_) ||
            !std::isfinite(max_steering_) ||
            min_steering_ > max_steering_)
        {
            RCLCPP_WARN(
                this->get_logger(),
                "Invalid steering bounds; using [-0.58, 0.58]");
            min_steering_ = -0.58;
            max_steering_ = 0.58;
        }
    }

    void nav2_command_callback(const geometry_msgs::msg::Twist & command)
    {
        const double speed = command.linear.x;
        const double steering = command.angular.z;

        if (!std::isfinite(speed) || !std::isfinite(steering)) {
            {
                const std::lock_guard<std::mutex> lock(state_mutex_);
                requested_speed_ = 0.0;
                requested_steering_ = 0.0;
                has_valid_command_ = false;
            }

            RCLCPP_ERROR_THROTTLE(
                this->get_logger(),
                *this->get_clock(),
                2000,
                "Rejected non-finite /cmd_vel_nav command; forcing stop");
            return;
        }

        {
            const std::lock_guard<std::mutex> lock(state_mutex_);
            requested_speed_ = clamp(speed, min_speed_, max_speed_);
            requested_steering_ = clamp(
                steering_scale_ * steering + steering_offset_,
                min_steering_,
                max_steering_);
            last_command_time_ = std::chrono::steady_clock::now();
            has_valid_command_ = true;
        }
    }

    bool publisher_conflict_detected()
    {
        const std::size_t publisher_count =
            this->count_publishers("/cmd_vel_nav");

        if (publisher_count > 1U) {
            bool first_conflict = false;
            {
                const std::lock_guard<std::mutex> lock(state_mutex_);
                first_conflict = !publisher_conflict_active_;
                publisher_conflict_active_ = true;
                has_valid_command_ = false;
            }

            if (first_conflict) {
                RCLCPP_ERROR(
                    this->get_logger(),
                    "Detected %zu publishers on /cmd_vel_nav; forcing stop until "
                    "only one command source remains",
                    publisher_count);
            } else {
                RCLCPP_WARN_THROTTLE(
                    this->get_logger(),
                    *this->get_clock(),
                    2000,
                    "%zu publishers remain on /cmd_vel_nav; output is stopped",
                    publisher_count);
            }

            return true;
        }

        bool conflict_cleared = false;
        {
            const std::lock_guard<std::mutex> lock(state_mutex_);
            conflict_cleared = publisher_conflict_active_;
            publisher_conflict_active_ = false;
        }

        if (conflict_cleared) {
            RCLCPP_WARN(
                this->get_logger(),
                "Publisher conflict cleared; waiting for a new valid command");
        }

        return false;
    }

    void command_publish()
    {
        const bool conflict = publisher_conflict_detected();
        bool timed_out = false;
        double command_age = 0.0;
        double steering_to_publish = 0.0;
        double speed_to_publish = 0.0;

        {
            const std::lock_guard<std::mutex> lock(state_mutex_);
            bool use_command = !conflict && has_valid_command_;

            if (use_command) {
                const auto now = std::chrono::steady_clock::now();
                command_age =
                    std::chrono::duration<double>(now - last_command_time_).count();

                if (command_age > command_timeout_sec_) {
                    use_command = false;
                    has_valid_command_ = false;
                    timed_out = true;
                }
            }

            if (use_command) {
                output_speed_ = requested_speed_;
                output_steering_ = requested_steering_;
            } else {
                output_speed_ = 0.0;
                output_steering_ = 0.0;
            }

            speed_to_publish = output_speed_;
            steering_to_publish = output_steering_;
        }

        if (timed_out) {
            RCLCPP_WARN(
                this->get_logger(),
                "/cmd_vel_nav command timed out after %.3f seconds; forcing stop",
                command_age);
        }

        publish_motor_command(steering_to_publish, speed_to_publish);
    }

    void publish_motor_command(double steering, double speed)
    {
        qcar2_interfaces::msg::MotorCommands motor_command;
        motor_command.motor_names = {"steering_angle", "motor_throttle"};
        motor_command.values = {steering, speed};
        const std::lock_guard<std::mutex> lock(publish_mutex_);
        command_publisher_->publish(motor_command);
    }

    void led_publish()
    {
        double output_speed = 0.0;
        double output_steering = 0.0;
        {
            const std::lock_guard<std::mutex> lock(state_mutex_);
            output_speed = output_speed_;
            output_steering = output_steering_;
        }

        for (bool & led_value : led_values_) {
            led_value = false;
        }

        if (output_speed != 0.0) {
            for (int i = 8; i <= 13; ++i) {
                led_values_[i] = true;
            }

            if (output_steering > 0.45) {
                led_values_[14] = true;
                led_values_[6] = true;
            } else if (output_steering < -0.45) {
                led_values_[15] = true;
                led_values_[7] = true;
            }
        } else {
            for (int i = 0; i <= 3; ++i) {
                led_values_[i] = true;
            }
        }

        qcar2_interfaces::msg::BooleanLeds led_commands;
        led_commands.led_names = {
            "left_outside_brake_light",
            "left_inside_brake_light",
            "right_inside_brake_light",
            "right_outside_brake_light",
            "left_reverse_light",
            "right_reverse_light",
            "left_rear_signal",
            "right_rear_signal",
            "left_outside_headlight",
            "left_middle_headlight",
            "left_inside_headlight",
            "right_inside_headlight",
            "right_middle_headlight",
            "right_outside_headlight",
            "left_front_signal",
            "right_front_signal"};
        led_commands.values.assign(std::begin(led_values_), std::end(led_values_));
        led_publisher_->publish(led_commands);
    }

    bool led_values_[16] = {false};

    double requested_speed_ = 0.0;
    double requested_steering_ = 0.0;
    double output_speed_ = 0.0;
    double output_steering_ = 0.0;

    double command_timeout_sec_ = 0.25;
    int64_t publish_period_ms_ = 20;
    double steering_scale_ = 1.0;
    double steering_offset_ = 0.0;
    double min_speed_ = -0.60;
    double max_speed_ = 0.60;
    double min_steering_ = -0.58;
    double max_steering_ = 0.58;

    bool has_valid_command_ = false;
    bool publisher_conflict_active_ = false;
    std::chrono::steady_clock::time_point last_command_time_;
    std::mutex state_mutex_;
    std::mutex publish_mutex_;

    rclcpp::TimerBase::SharedPtr command_timer_;
    rclcpp::TimerBase::SharedPtr led_timer_;
    rclcpp::Publisher<qcar2_interfaces::msg::MotorCommands>::SharedPtr
        command_publisher_;
    rclcpp::Publisher<qcar2_interfaces::msg::BooleanLeds>::SharedPtr
        led_publisher_;
    rclcpp::Subscription<geometry_msgs::msg::Twist>::SharedPtr nav2_subscriber_;
};


int main(int argc, char ** argv)
{
    rclcpp::init(argc, argv);

    auto node = std::make_shared<Nav2QCarConverter>();
    const std::weak_ptr<Nav2QCarConverter> weak_node(node);

    rclcpp::on_shutdown([weak_node]() {
        if (const auto converter = weak_node.lock()) {
            converter->publish_stop_best_effort();
        }
    });

    rclcpp::spin(node);
    node->publish_stop_best_effort();
    node.reset();

    if (rclcpp::ok()) {
        rclcpp::shutdown();
    }

    return 0;
}
