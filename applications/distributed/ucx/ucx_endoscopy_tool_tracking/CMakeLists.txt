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

cmake_minimum_required(VERSION 3.20)
project(ucx_endoscopy_tool_tracking_apps LANGUAGES NONE)

# Default to download datasets
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)

# Download the endoscopy sample data
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  holoscan_download_data(endoscopy
    URL nvidia/clara-holoscan/holoscan_endoscopy_sample_data:20230222
    DOWNLOAD_NAME holoscan_endoscopy_sample_data_20230222.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 480
    GXF_ENTITIES_WIDTH 854
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
  )
endif()


# Add testing
if(BUILD_TESTING)
  file(MAKE_DIRECTORY ${HOLOHUB_DATA_DIR}/endoscopy/engines)
  add_test(
    NAME endoscopy_engine_generation
    COMMAND python3 ${CMAKE_SOURCE_DIR}/utilities/generate_trt_engine.py
      --input ${HOLOHUB_DATA_DIR}/endoscopy/tool_loc_convlstm.onnx
      --output ${HOLOHUB_DATA_DIR}/endoscopy/engines/
      --fp16
  )
  set_tests_properties(endoscopy_engine_generation PROPERTIES FIXTURES_SETUP endoscopy_engine_generation_fixture)
endif()

add_subdirectory(cpp)
add_subdirectory(python)
