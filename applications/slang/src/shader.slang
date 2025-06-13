/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

import holoscan;

[holoscan::input("input_buffer")]
StructuredBuffer<int> input_buffer;

[holoscan::output("output_buffer")]
[holoscan::size_of("input_buffer")]
RWStructuredBuffer<int> output_buffer;

[holoscan::parameter("parameter")]
uniform int parameter;

[holoscan::grid_size_of("input_buffer.Count")]
[shader("compute")]
void main_dynamic(uint3 gid : SV_DispatchThreadID)
{
   output_buffer[gid.x] = input_buffer[gid.x] + parameter.x;
}
