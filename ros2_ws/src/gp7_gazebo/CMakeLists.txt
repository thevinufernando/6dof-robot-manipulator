cmake_minimum_required(VERSION 3.8)
project(gp7_gazebo)

find_package(ament_cmake REQUIRED)

install(DIRECTORY config launch models worlds
  DESTINATION share/${PROJECT_NAME})

install(PROGRAMS scripts/move_box_b.py
  DESTINATION lib/${PROJECT_NAME})

ament_package()
