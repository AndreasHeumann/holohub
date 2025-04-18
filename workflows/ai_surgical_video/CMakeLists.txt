# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
project(ai_surgical_video_workflow LANGUAGES NONE)

# Holoscan
find_package(holoscan 3.0 REQUIRED CONFIG
             PATHS "/opt/nvidia/holoscan" "/workspace/holoscan-sdk/install")

# Default to download datasets
option(HOLOHUB_DOWNLOAD_DATASETS "Download datasets" ON)

# Download the required sample data and models
if(HOLOHUB_DOWNLOAD_DATASETS)
  include(holoscan_download_data)
  # Orsi partial nephrectomy procedures and out-of-body detection model
  holoscan_download_data(orsi
    URL  nvidia/clara-holoscan/holoscan_orsi_academy_sample_data:20240206
    DOWNLOAD_NAME orsi.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    GENERATE_GXF_ENTITIES
    GXF_ENTITIES_HEIGHT 1080
    GXF_ENTITIES_WIDTH 1920
    GXF_ENTITIES_CHANNELS 3
    GXF_ENTITIES_FRAMERATE 30
    ALL
  )
  # SSD detection for endoscopy surgical tools
  holoscan_download_data(ssd_model
    URL nvidia/clara-holoscan/ssd_surgical_tool_detection_model:v0.1
    DOWNLOAD_NAME ssd_surgical_tool_detection_model_v0.1.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    ALL
  )
  # MONAI endoscopic tool segmentation model
  holoscan_download_data(monai_tool_seg_model
    URL nvidia/clara-holoscan/monai_endoscopic_tool_segmentation_model:v0.1
    DOWNLOAD_NAME monai_endoscopic_tool_segmentation_model_v0.1.zip
    DOWNLOAD_DIR ${HOLOHUB_DATA_DIR}
    ALL
  )
endif()
