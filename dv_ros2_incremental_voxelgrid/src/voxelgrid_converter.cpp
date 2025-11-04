#include "voxelgrid_converter.hpp"

namespace voxelgrid_converter
{
    VoxelgridConverter::VoxelgridConverter(const rclcpp::NodeOptions &options)
        : Node("voxelgrid_converter", options), m_node{this}
    {
        RCLCPP_INFO(m_node->get_logger(), "[VoxelgridConverter::VoxelgridConverter] Initializing...");
        auto node_name = m_node->get_name();
        this->voxelgrid_converter_ctor(node_name);
        RCLCPP_INFO(m_node->get_logger(), "[VoxelgridConverter::VoxelgridConverter] Initialization complete!");
        RCLCPP_INFO(m_node->get_logger(), "[VoxelgridConverter::VoxelgridConverter] Beginning spin...");
        this->start();
    }

    void VoxelgridConverter::voxelgrid_converter_ctor(const std::string &t_node_name)
    {
        RCLCPP_INFO(m_node->get_logger(), "Constructor is initialized for node: %s", t_node_name.c_str());
        parameterInitilization();
        if(!readParameters())
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read parameters.");
            rclcpp::shutdown();
            std::exit(EXIT_FAILURE);
        }
        parameterPrinter();

        m_slicer = std::make_unique<dv::EventStreamSlicer>();

        m_events_tensor_cpu = torch::zeros({m_params.events_kept, 4}, torch::kFloat32).to(torch::kCPU).contiguous();

        std::cout << "Event Tensor shape: " << m_events_tensor_cpu.sizes() << std::endl;

        // Initialize the time bin ctr to zero
        time_bin_ctr = 0;

        // Initialize event frame size from parameters
        ev_width = static_cast<float>(m_params.ev_width);
        ev_height = static_cast<float>(m_params.ev_height);
        voxelgrid_time_bins = m_params.time_bins;
        microseconds_per_bin = 1e6 / m_params.frame_rate;

        // Initialize the voxel grid tensor
        m_voxelgrid_tensor = torch::zeros(
            {voxelgrid_time_bins, m_params.ev_height, m_params.ev_width},
            torch::kFloat32
        ).to(torch::kCPU).contiguous();

        m_events_subscriber = m_node->
            create_subscription<dv_ros2_msgs::msg::EventArray>(
                m_params.input_topic,
                10,
                std::bind(
                    &VoxelgridConverter::eventCallback,
                    this,
                    std::placeholders::_1
                )
            );

        m_voxelgrid_publisher = m_node->
            create_publisher<std_msgs::msg::Float32MultiArray>(m_params.output_topic, 10);

        RCLCPP_INFO(m_node->get_logger(), "Sucessfully launched.");
    }

    void VoxelgridConverter::start()
    {
        // start slicer
        startSlicer();

        // run thread
        m_spin_thread = true;
        m_voxelgrid_thread = std::thread(&VoxelgridConverter::build_voxelgrids, this);
        RCLCPP_INFO(m_node->get_logger(), "Inference thread is started.");
    }

    void VoxelgridConverter::stop()
    {
        RCLCPP_INFO(m_node->get_logger(), "Stopping the inference thread...");

        // stop the thread first
        if (m_spin_thread)
        {
            m_spin_thread = false;
            m_voxelgrid_thread.join();
        }
    }

    bool VoxelgridConverter::isRunning() const
    {
        return m_spin_thread.load(std::memory_order_relaxed);
    }

    void VoxelgridConverter::eventCallback(dv_ros2_msgs::msg::EventArray::SharedPtr events)
    {
        auto store = dv_ros2_msgs::toEventStore(*events);
        try
        {
            m_slicer->accept(store);
        }
        catch (std::out_of_range &e)
        {
            RCLCPP_WARN_STREAM(m_node->get_logger(), "Event out of range: " << e.what());
        }
    }

    void VoxelgridConverter::slicerCallback(const dv::EventStore &events)
    {
        m_event_queue.push(events);
    }

    void VoxelgridConverter::updateConfiguration()
    {
        if (m_job_id.has_value())
        {
            m_slicer->removeJob(m_job_id.value());
        }
        startSlicer();
    }

    void VoxelgridConverter::startSlicer()
    {
        // convert frame_rate to ms (delta time)
        int32_t delta_time = static_cast<int>(1000 / m_params.frame_rate);
        m_job_id = m_slicer->
            doEveryTimeInterval(
                dv::Duration(delta_time * 1000LL),
                std::bind(&VoxelgridConverter::slicerCallback, this, std::placeholders::_1)
            );
    }

    int32_t VoxelgridConverter::eventStoreToTensor(const dv::EventStore &events)
    {
        float step = 1.0f;
        int64_t const N = events.size();
        int const usable_events = m_params.events_kept;

        // subsample events to fit into max_events
        if (N > usable_events)
            step = static_cast<float>(N) / static_cast<float>(usable_events);

        int64_t const m_timestamp_offset = events[0].timestamp();

        float64_t cnt = 0.0f; // original event index
        int32_t events_filled = 0; // tensor row index

        try
        {
        auto accessor = m_events_tensor_cpu.accessor<float, 2>();
        while (events_filled < usable_events)
        {
            int64_t const cnt_int = static_cast<int64_t>(cnt);
            if (cnt_int >= N)
                break;

            float const normalized_time = (events[cnt_int].timestamp() - m_timestamp_offset) / microseconds_per_bin;

            if (normalized_time >= 1.0f)
                break;

            // direct memory access for speed using accessor
            accessor[events_filled][0] = static_cast<float>(events[cnt_int].x());
            accessor[events_filled][1] = static_cast<float>(events[cnt_int].y());
            accessor[events_filled][2] = normalized_time;
            accessor[events_filled][3] = static_cast<float>(events[cnt_int].polarity() ? 1.0f : -1.0f);

            // next event
            cnt += step;
            events_filled++;
        }
        }
        catch (const c10::Error &e)
        {
            RCLCPP_ERROR(m_node->get_logger(), "LibTorch error during event transfer to tensor: %s", e.what());
            return -1;
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(m_node->get_logger(), "Error during event transfer to tensor: %s", e.what());
            return -1;
        }

        return events_filled;
    }

    void VoxelgridConverter::build_voxelgrids()
    {
        if (m_spin_thread == false)
        {
            RCLCPP_WARN(m_node->get_logger(), "Inference thread started while spin_thread is false.");
            return;
        }
        while (m_spin_thread)
        {
            m_event_queue.consume_all([&](const dv::EventStore &events)
            {
                int const N = events.size();
                if (N == 0)
                {
                    RCLCPP_WARN(m_node->get_logger(), "No events to process in execute_inference.");
                    return;
                }

                RCLCPP_INFO(m_node->get_logger(), "Processing %d events in execute_inference.", N);
                int32_t const events_filled = eventStoreToTensor(events);
                RCLCPP_INFO(m_node->get_logger(), "Filled %d events into tensor.", events_filled);

                torch::Tensor sliced_events_tensor_cpu = m_events_tensor_cpu.index({
                    torch::indexing::Slice(torch::indexing::None, events_filled)
                });

                update_voxelgrid(sliced_events_tensor_cpu);
                time_bin_ctr++;

                if (time_bin_ctr == voxelgrid_time_bins - 1)
                {
                    publish_voxelgrid();
                    // Reset voxelgrid tensor and time bin counter
                    m_voxelgrid_tensor.zero_();
                    time_bin_ctr = 0;
                }
            });
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }

    void VoxelgridConverter::update_voxelgrid(torch::Tensor &events_tensor)
    {
        /* events_tensor shape: [N, 4] where each row is (x, y, t, p)
           x: pixel x coordinate float
           y: pixel y coordinate float
           t: normalized time in [0, 1)
           p: polarity (+1 or -1)
        */
        torch::Tensor x = events_tensor.index({torch::indexing::Slice(), 0}).to(torch::kInt32);
        torch::Tensor y = events_tensor.index({torch::indexing::Slice(), 1}).to(torch::kInt32);
        torch::Tensor t = events_tensor.index({torch::indexing::Slice(), 2}).to(torch::kFloat32);
        torch::Tensor p = events_tensor.index({torch::indexing::Slice(), 3}).to(torch::kFloat32);

        // We assume x,y are ints. Might not be true for undistorted events.
        torch::Tensor weight = p * (1.0f - t);
        torch::Tensor t_indices = torch::full_like(x, time_bin_ctr, torch::kInt32);

        m_voxelgrid_tensor.index_put_({t_indices, y, x}, weight, true);
        m_voxelgrid_tensor.index_put_({t_indices + 1, y, x}, p - weight, true);
    }

    void VoxelgridConverter::publish_voxelgrid()
    {
        // Create Float32MultiArray message
        auto msg = std_msgs::msg::Float32MultiArray();

        // Set dimensions: [time_bins, height, width]
        msg.layout.dim.resize(3);
        msg.layout.dim[0].label = "time_bins";
        msg.layout.dim[0].size = voxelgrid_time_bins;
        msg.layout.dim[0].stride = voxelgrid_time_bins * m_params.ev_height * m_params.ev_width;

        msg.layout.dim[1].label = "height";
        msg.layout.dim[1].size = m_params.ev_height;
        msg.layout.dim[1].stride = m_params.ev_height * m_params.ev_width;

        msg.layout.dim[2].label = "width";
        msg.layout.dim[2].size = m_params.ev_width;
        msg.layout.dim[2].stride = m_params.ev_width;

        msg.layout.data_offset = 0;

        float* data_ptr = m_voxelgrid_tensor.data_ptr<float>();

        int64_t total_size = voxelgrid_time_bins * m_params.ev_height * m_params.ev_width;

        msg.data.assign(data_ptr, data_ptr + total_size);

        m_voxelgrid_publisher->publish(msg);

        RCLCPP_INFO(m_node->get_logger(), "Published voxelgrid with shape [%d, %d, %d]",
                    voxelgrid_time_bins, m_params.ev_height, m_params.ev_width);
    }

    VoxelgridConverter::~VoxelgridConverter()
    {
        RCLCPP_INFO(m_node->get_logger(), "Destructor is activated.");
        stop();
        rclcpp::shutdown();
    }

    inline void VoxelgridConverter::parameterInitilization() const
    {
        rcl_interfaces::msg::ParameterDescriptor descriptor;
        rcl_interfaces::msg::IntegerRange int_range;
        rcl_interfaces::msg::FloatingPointRange float_range;

        m_node->declare_parameter("input_topic", m_params.input_topic);
        m_node->declare_parameter("output_topic", m_params.output_topic);

        float_range.set__from_value(10.0).set__to_value(1000.0);
        descriptor.floating_point_range = {float_range};
        m_node->declare_parameter("frame_rate", m_params.frame_rate, descriptor);
        int_range.set__from_value(10'000).set__to_value(200'000).set__step(1);
        descriptor.integer_range = {int_range};
        m_node->declare_parameter("events_kept", m_params.events_kept, descriptor);
        m_node->declare_parameter("ev_width", m_params.ev_width);
        m_node->declare_parameter("ev_height", m_params.ev_height);
        m_node->declare_parameter("time_bins", m_params.time_bins);
    }

    inline void VoxelgridConverter::parameterPrinter() const
    {
        RCLCPP_INFO(m_node->get_logger(), "-------- Parameters --------");
        RCLCPP_INFO(m_node->get_logger(), "input_topic: %s", m_params.input_topic.c_str());
        RCLCPP_INFO(m_node->get_logger(), "output_topic: %s", m_params.output_topic.c_str());
        RCLCPP_INFO(m_node->get_logger(), "frame_rate: %f", m_params.frame_rate);
        RCLCPP_INFO(m_node->get_logger(), "events_kept: %d", m_params.events_kept);
        RCLCPP_INFO(m_node->get_logger(), "ev_width: %d", m_params.ev_width);
        RCLCPP_INFO(m_node->get_logger(), "ev_height: %d", m_params.ev_height);
        RCLCPP_INFO(m_node->get_logger(), "time_bins: %d", m_params.time_bins);
        RCLCPP_INFO(m_node->get_logger(), "----------------------------");
    }

    inline bool VoxelgridConverter::readParameters()
    {
        if (!m_node->get_parameter("input_topic", m_params.input_topic))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter input_topic.");
            return false;
        }
        if (!m_node->get_parameter("output_topic", m_params.output_topic))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter output_topic.");
            return false;
        }
        if (!m_node->get_parameter("frame_rate", m_params.frame_rate))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter frame_rate.");
            return false;
        }
        if (!m_node->get_parameter("events_kept", m_params.events_kept))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter events_kept.");
            return false;
        }
        if (!m_node->get_parameter("ev_width", m_params.ev_width))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter ev_width.");
            return false;
        }
        if (!m_node->get_parameter("ev_height", m_params.ev_height))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter ev_height.");
            return false;
        }
        if (!m_node->get_parameter("time_bins", m_params.time_bins))
        {
            RCLCPP_ERROR(m_node->get_logger(), "Failed to read paramter time_bins.");
            return false;
        }
        return true;
    }

    rcl_interfaces::msg::SetParametersResult VoxelgridConverter::paramsCallback(const std::vector<rclcpp::Parameter> &parameters)
    {
        rcl_interfaces::msg::SetParametersResult result;
        result.successful = true;
        result.reason = "success";
        
        for (const auto &param : parameters)
        {
            if (param.get_name() == "input_topic")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
                {
                    m_params.input_topic = param.as_string();
                }
                else
                {
                    result.successful = false;
                    result.reason = "input_topic must be a string";
                }
            }
            else if (param.get_name() == "output_topic")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_STRING)
                {
                    m_params.output_topic = param.as_string();
                }
                else
                {
                    result.successful = false;
                    result.reason = "output_topic must be a string";
                }
            }
            else if (param.get_name() == "frame_rate")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_DOUBLE)
                {
                    m_params.frame_rate = param.as_double();
                }
                else
                {
                    result.successful = false;
                    result.reason = "frame_rate must be a double";
                }
            }
            else if (param.get_name() == "events_kept")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
                {
                    m_params.events_kept = param.as_int();
                }
                else
                {
                    result.successful = false;
                    result.reason = "events_kept must be an integer";
                }
            }
            else if (param.get_name() == "ev_width")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
                {
                    m_params.ev_width = param.as_int();
                }
                else
                {
                    result.successful = false;
                    result.reason = "ev_width must be an integer";
                }
            }
            else if (param.get_name() == "ev_height")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
                {
                    m_params.ev_height = param.as_int();
                }
                else
                {
                    result.successful = false;
                    result.reason = "ev_height must be an integer";
                }
            }
            else if (param.get_name() == "time_bins")
            {
                if (param.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER)
                {
                    m_params.time_bins = param.as_int();
                }
                else
                {
                    result.successful = false;
                    result.reason = "time_bins must be an integer";
                }
            }
            else
            {
                result.successful = false;
                result.reason = "unknown parameter";
            }
        }
        updateConfiguration();
        return result;
    }


} // namespace voxelgrid_converter


#include "rclcpp_components/register_node_macro.hpp"
RCLCPP_COMPONENTS_REGISTER_NODE(voxelgrid_converter::VoxelgridConverter)
