cmake_minimum_required(VERSION 3.8)
project(haiying_v9_2)

find_package(ament_cmake REQUIRED)

install(
  DIRECTORY
    launch
    models
    urdf
    worlds
  DESTINATION share/${PROJECT_NAME}
)

install(
  PROGRAMS
    scripts/publish_v9_2_model.py
    scripts/publish_world_base_tf.py
  DESTINATION lib/${PROJECT_NAME}
)

ament_package()
