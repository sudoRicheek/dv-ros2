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

// Boost Headers
#include <boost/lockfree/spsc_queue.hpp>

#include "rclcpp/rclcpp.hpp"

// dv_ros2_msgs Headers
#include "dv_ros2_msgs/msg/event.hpp"
#include "dv_ros2_msgs/msg/event_array.hpp"
#include "dv_ros2_messaging/messaging.hpp"

// std_msgs Headers
#include "std_msgs/msg/float32_multi_array.hpp"

#include <torch/torch.h>


namespace voxelgrid_converter
{
    struct Params
    {
        /// @brief topic name of the input event array
        std::string input_topic = "events";

        /// @brief topic name of the output inference results
        std::string output_topic = "voxelgrid";

        /// @brief size of the event frame to be processed
        int ev_width = 640;
        int ev_height = 480;
        float frame_rate = 20.0; // in Hz
        int events_kept = 200'000; // number of events to keep for inference

        /// @brief voxelgrid params
        int time_bins = 5; // take in these many time bins of events arriving at frame_rate
        // So The output voxelgrid will sumamrize (time_bins - 1) * (1/frame_rate) seconds of events, here 200ms.
    };

    class VoxelgridConverter : public rclcpp::Node
    {
        using rclcpp::Node::Node;
    public:
        explicit VoxelgridConverter(const rclcpp::NodeOptions &options);
        void voxelgrid_converter_ctor(const std::string &t_node_name);
        ~VoxelgridConverter();
        void start();
        void stop();
        bool isRunning() const;
        rcl_interfaces::msg::SetParametersResult paramsCallback(const std::vector<rclcpp::Parameter> &parameters);

    private:
        /// @brief Width, height, and time of the event frame, grabbed from parameters. Float becuase we use them for normalization.
        float ev_width = 0;
        float ev_height = 0;

        /// @brief Voxelgrid time bins
        int voxelgrid_time_bins;

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

        /// @brief Converts an EventStore to a tensor
        int32_t eventStoreToTensor(const dv::EventStore &event_store);

        /// @brief Inference execution thread
        void build_voxelgrids();

        /// @brief updates the Voxelgrid tensor with a event tensor
        void update_voxelgrid(torch::Tensor &events_tensor);

        /// @brief Publishes the voxelgrid tensor as a ROS2 message
        void publish_voxelgrid();

        /// @brief Starts the slicer job
        void startSlicer();

        /// @brief To keep track of time bins filled in the voxelgrid. It is published when it reaches time_bins.
        int time_bin_ctr;

        /// @brief frame rate in us
        float microseconds_per_bin;

        /// @brief rclcpp node pointer
        rclcpp::Node::SharedPtr m_node;

        /// @brief Parameters
        Params m_params;

        // Thread related
        std::atomic<bool> m_spin_thread = true;
        std::thread m_voxelgrid_thread;

        /// @brief Tensor to hold events on CPU
        torch::Tensor m_events_tensor_cpu;

        /// @brief Tensor to hold voxelgrid on CPU
        torch::Tensor m_voxelgrid_tensor;

        /// @brief EventArray subscriber
        rclcpp::Subscription<dv_ros2_msgs::msg::EventArray>::SharedPtr m_events_subscriber;

        /// @brief Voxelgrid publisher
        rclcpp::Publisher<std_msgs::msg::Float32MultiArray>::SharedPtr m_voxelgrid_publisher;

        /// @brief boost SPSC queue for event storage between callback and inference thread
        boost::lockfree::spsc_queue<dv::EventStore> m_event_queue{100};

        /// @brief Slicer object
        std::unique_ptr<dv::EventStreamSlicer> m_slicer = nullptr;

        /// @brief Job ID of the slicer, used to stop jobs running in the slicer
        std::optional<int> m_job_id;
    };
} // end namespace voxelgrid_converter
