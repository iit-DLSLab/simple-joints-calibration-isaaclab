cmake_minimum_required(VERSION 2.8.3)
project(dls_msgs)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  geometry_msgs
  std_msgs
  message_generation
  actionlib_msgs
  )

add_message_files(
    FILES
    rl_signal_in.msg
    rl_signal_out.msg
    )


generate_messages(
   DEPENDENCIES
   std_msgs
   geometry_msgs
   actionlib_msgs
   )

catkin_package(
  CATKIN_DEPENDS message_runtime
  DEPENDS )

