cmake_minimum_required(VERSION 3.8)
project(robot_driver)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 查找依赖
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(robot_interfaces REQUIRED)

# 2. 包含 SCServo 头文件路径
include_directories(include)
include_directories(src) # 因为你的 SCServo 文件夹在 src 下

# 3. 收集 SCServo 的源文件 (SCS.cpp, SMS_STS.cpp 等)
file(GLOB_RECURSE SERVO_SOURCES "src/SCServo/*.cpp")

# 4. 定义节点可执行文件
add_executable(driver_node 
  src/driver_node.cpp
  ${SERVO_SOURCES}  # 把库文件一起编译
)

# 5. 链接依赖
ament_target_dependencies(driver_node
  rclcpp
  robot_interfaces
)

# 6. 安装规则
install(TARGETS
  driver_node
  DESTINATION lib/${PROJECT_NAME}
)

# 记得把 SCServo 文件夹也作为资源安装（可选，但推荐）
install(DIRECTORY src/SCServo
  DESTINATION share/${PROJECT_NAME}/src
)

ament_package()