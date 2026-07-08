cmake_minimum_required(VERSION 3.16)
project(sim2real_bridge_cpp)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)

set(DEFAULT_ONNXRUNTIME_ROOT
  "${CMAKE_CURRENT_SOURCE_DIR}/../../thirdparty/onnxruntime-linux-x64-1.22.0")
set(ONNXRUNTIME_ROOT "${DEFAULT_ONNXRUNTIME_ROOT}" CACHE PATH "ONNX Runtime root")
get_filename_component(ONNXRUNTIME_ROOT "${ONNXRUNTIME_ROOT}" ABSOLUTE)
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_ROOT}/include")
set(ONNXRUNTIME_LIB "${ONNXRUNTIME_ROOT}/lib/libonnxruntime.so.1.22.0")

if(NOT EXISTS "${ONNXRUNTIME_LIB}")
  message(FATAL_ERROR
    "ONNX Runtime library not found: ${ONNXRUNTIME_LIB}. "
    "Set -DONNXRUNTIME_ROOT=/path/to/onnxruntime-linux-x64-1.22.0")
endif()

add_executable(a1_policy_bridge_cpp src/a1_policy_bridge_cpp.cpp)
target_compile_features(a1_policy_bridge_cpp PRIVATE cxx_std_17)
target_include_directories(a1_policy_bridge_cpp PRIVATE
  include
  "${ONNXRUNTIME_INCLUDE_DIR}"
)
target_link_libraries(a1_policy_bridge_cpp "${ONNXRUNTIME_LIB}")
ament_target_dependencies(a1_policy_bridge_cpp
  rclcpp
  sensor_msgs
  std_msgs
)

set_target_properties(a1_policy_bridge_cpp PROPERTIES
  BUILD_RPATH "${ONNXRUNTIME_ROOT}/lib"
  INSTALL_RPATH "${ONNXRUNTIME_ROOT}/lib"
)

install(TARGETS a1_policy_bridge_cpp
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

ament_package()
