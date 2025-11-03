#pragma once

#include <thread>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <numeric>
#include <memory>
#include <tuple>

#include "spsc_value.hpp"
#include "rclcpp/rclcpp.hpp"

// dv_ros2_msgs Headers
#include "dv_ros2_msgs/msg/event.hpp"
#include "dv_ros2_msgs/msg/event_array.hpp"
#include "dv_ros2_messaging/messaging.hpp"

// sensor_msgs Headers
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>

#include <torch/torch.h>
#include <torch/csrc/inductor/aoti_package/model_package_loader.h>

namespace torch_converter
{
    struct Params
    {
        /// @brief topic name of the input event array
        std::string input_topic = "events";
        /// @brief topic name of the output inference results
        std::string output_topic = "depth";
        /// @brief path to the pt2 model file
        std::string f3_pt2_path = "";
        std::string dav2_pt2_path = "";
        /// @brief size of the event frame to be processed
        int ev_width = 640;
        int ev_height = 480;
        int ev_time = 20; // in ms
        float frame_rate = 50.0; // in Hz
        int events_kept = 50'000; // number of events to keep for inference
        int dav2_height = 238; // default DAV2 height to run inference
        int dav2_width = 308; // default DAV2 width to run inference
    };

    class TorchConverter : public rclcpp::Node
    {
        using rclcpp::Node::Node;
    public:
        explicit TorchConverter(const rclcpp::NodeOptions &options);
        void torch_converter_ctor(const std::string &t_node_name);
        ~TorchConverter();
        void start();
        void stop();
        bool isRunning() const;
        rcl_interfaces::msg::SetParametersResult paramsCallback(const std::vector<rclcpp::Parameter> &parameters);
    private:

        /// @brief Width, height, and time of the event frame, grabbed from parameters. Float becuase we use them for normalization.
        float ev_width = 0;
        float ev_height = 0;
        float ev_time = 0;

        /// @brief Parameter initialization
        inline void parameterInitilization() const;

        /// @brief Print parameters
        inline void parameterPrinter() const;

        /// @brief Reads the std library variables and ROS2 parameters
        /// @return true if all parameters are read successfully
        inline bool readParameters();

        /// @brief Update configuration for reconfiguration while running
        void updateConfiguration();

        /// @brief Event callback function for populating queue
        /// @param events EventArray message
        void eventCallback(dv_ros2_msgs::msg::EventArray::SharedPtr events);

        /// @brief Slicer callback function
        void slicerCallback(const dv::EventStore &events);

        /// @brief Inference execution thread
        void execute_inference();

        int32_t eventStoreToTensor(const dv::EventStore &event_store);

        void publish_depth_image(torch::Tensor &depth_tensor);

        void startSlicer();

        long unsigned int m_frame_id = 0;

        /// @brief rclcpp node pointer
        rclcpp::Node::SharedPtr m_node;

        /// @brief Parameters
        Params m_params;

        // Thread related
        std::atomic<bool> m_spin_thread = true;
        std::thread m_inference_thread;

        torch::Tensor m_events_tensor_cuda;

        torch::Tensor m_events_tensor_cpu;

        /// @brief EventArray subscriber
        rclcpp::Subscription<dv_ros2_msgs::msg::EventArray>::SharedPtr m_events_subscriber;

        /// @brief Inference result publisher
        rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr m_depth_publisher;

        /// @brief Event spsc value to store the latest event store. This is a triple buffer
        /// implementation adapted from boost lockfree spsc value 1.88, since we have 
        /// an older boost version in boost-invation.
        boost::lockfree::spsc_value<dv::EventStore> m_event_value;

        /// @brief Slicer object
        std::unique_ptr<dv::EventStreamSlicer> m_slicer = nullptr;

        /// @brief Job ID of the slicer, used to stop jobs running in the slicer
        std::optional<int> m_job_id;

        /// @brief Inference engine F3 loader
        std::optional<torch::inductor::AOTIModelPackageLoader> f3_loader;

        /// @brief Inference engine DAV2 loader
        std::optional<torch::inductor::AOTIModelPackageLoader> dav2_loader;
    };
} // end namespace torch_converter
