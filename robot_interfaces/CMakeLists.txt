cmake_minimum_required(VERSION 3.8)
project(robot_interfaces)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED) # VisionState 引用了 Header，需要这个

# 生成接口
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/MotorCommand.msg"
  "msg/MotorState.msg"
  "msg/VisionState.msg"  # <--- 确保这一行准确无误
  DEPENDENCIES std_msgs  # <--- 因为用了 Header，必须加依赖
)

ament_package()