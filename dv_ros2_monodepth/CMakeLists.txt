cmake_minimum_required(VERSION 3.8)
project(dv_ros2_monodepth)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- PyTorch C++ API configuration ---
execute_process(
    COMMAND python -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)


list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")


# find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(std_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(dv_ros2_msgs REQUIRED)
find_package(dv_ros2_messaging REQUIRED)
find_package(dv-processing REQUIRED)
find_package(Torch REQUIRED)

set(dependencies "std_msgs" "rclcpp" "geometry_msgs" "sensor_msgs" "dv_ros2_msgs" "dv_ros2_messaging" "rclcpp_components")

# --- CUDA configuration ---
set(CUDA 12.6)
set(CUDA_HOME /usr/local/cuda-${CUDA})
set(CUDA_PATH /usr/local/cuda-${CUDA})
set(NVCC /usr/local/cuda-$CUDA/bin/nvcc)

set(CMAKE_CXX_STANDARD 17)

include_directories(include())

# Add CUDA include directories manually
include_directories(${CUDA_HOME}/include)
include_directories(${CUDA_HOME}/targets/aarch64-linux/include)

# Include project-specific include directory
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include/")

# Optional: print paths for debugging
message(STATUS "Using CUDA from ${CUDA_HOME}")

file(GLOB_RECURSE SOURCES src/*.cpp)

### TORCH CONVERTER NODE COMPONENT ###

add_library(monodepth_component SHARED
  src/torch_converter.cpp
)
ament_target_dependencies(monodepth_component rclcpp rclcpp_components std_msgs sensor_msgs dv_ros2_messaging dv_ros2_msgs geometry_msgs)
rclcpp_components_register_nodes(monodepth_component "torch_converter::TorchConverter")

target_link_libraries(monodepth_component
  ${catkin_LIBRARIES}
  dv::processing
  "${TORCH_LIBRARIES}"
)
set_property(TARGET monodepth_component PROPERTY CXX_STANDARD 17)

install(TARGETS
  monodepth_component
  ARCHIVE DESTINATION lib
)

#################################

# TODO: check if config directory is required here
install(DIRECTORY
  launch
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  # the following line skips the linter which checks for copyrights
  # comment the line when a copyright and license is added to all source files
  set(ament_cmake_copyright_FOUND TRUE)
  # the following line skips cpplint (only works in a git repo)
  # comment the line when this package is in a git repo and when
  # a copyright and license is added to all source files
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
