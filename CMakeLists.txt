cmake_minimum_required(VERSION 3.0.2)
project(sustaina_ppo_walk_ros)

find_package(
  catkin REQUIRED
  COMPONENTS rospy
             std_msgs
)

catkin_package(
  CATKIN_DEPENDS
    rospy
    std_msgs
)
