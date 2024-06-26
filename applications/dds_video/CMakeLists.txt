# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.24)
project(dds_video)

find_package(holoscan 2.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

add_executable(dds_video dds_video.cpp)
target_link_libraries(dds_video
  holoscan::core
  holoscan::ops::holoviz
  holoscan::ops::v4l2
  holoscan::ops::dds_video_publisher
  holoscan::ops::dds_video_subscriber
  holoscan::ops::dds_shapes_subscriber
)

# Copy qos_profiles.xml to the binary directory
add_custom_target(dds_video_qos_profiles_xml
  COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/qos_profiles.xml" ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/qos_profiles.xml"
)
add_dependencies(dds_video dds_video_qos_profiles_xml)
