cmake_minimum_required(VERSION 3.8)
project(eat_nav_interfaces)
find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/DemFrame.msg"
  "msg/ActorState22.msg"
  "msg/CandidateBank.msg"
  "msg/GoalState.msg"
  "msg/CandidateEvaluation.msg"
  "msg/PathSelection.msg"
  "msg/NavigationStatus.msg"
  DEPENDENCIES builtin_interfaces std_msgs geometry_msgs)
ament_export_dependencies(rosidl_default_runtime)
ament_package()
