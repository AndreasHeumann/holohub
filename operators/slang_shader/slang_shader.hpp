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

#ifndef OPERATORS_SLANG_SHADER_SLANG_SHADER_HPP
#define OPERATORS_SLANG_SHADER_SLANG_SHADER_HPP

#include <memory>
#include <string>
#include <utility>

#include <holoscan/core/operator.hpp>

namespace holoscan::ops {

/**
 * @brief Slang operator.
 */
class SlangShaderOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_TEMPLATE()
  SlangShaderOp(const std::string& shader_source, ArgT&& arg, ArgsT&&... args)
      : Operator(std::forward<ArgT>(arg), std::forward<ArgsT>(args)...) {
    init(shader_source);
  }

  void setup(OperatorSpec& spec) override;

  void compute(InputContext& op_input, [[maybe_unused]] OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override;

 private:
  void init(const std::string& shader_source);

  // Forward declaration of implementation
  class Impl;
  std::shared_ptr<Impl> impl_;
};

}  // namespace holoscan::ops

#endif /* OPERATORS_SLANG_SHADER_SLANG_SHADER_HPP */
