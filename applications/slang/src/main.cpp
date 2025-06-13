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

#include <fstream>

#include <holoscan/core/application.hpp>
#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/resources/gxf/unbounded_allocator.hpp>

#include "slang_shader.hpp"

namespace holoscan::ops {

class SourceOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SourceOp)

  SourceOp() = default;

  void setup(OperatorSpec& spec) override {
    spec.param(allocator_,
               "allocator",
               "Allocator for output buffers.",
               "Allocator for output buffers.",
               std::static_pointer_cast<Allocator>(
                   fragment()->make_resource<UnboundedAllocator>("allocator")));
    spec.output<gxf::Entity>("output");
  }

  void compute(InputContext& op_input, OutputContext& op_output,
               ExecutionContext& context) override {
    auto entity = gxf::Entity::New(&context);
    auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>().value();
    // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
    auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
        context.context(),
        allocator_.has_value() ? allocator_.get()->gxf_cid()
                               : allocator_.default_value()->gxf_cid());
    tensor->reshape<int>(
        nvidia::gxf::Shape({1}), nvidia::gxf::MemoryStorageType::kHost, gxf_allocator.value());

    auto value = index_++;
    std::memcpy(tensor->pointer(), &value, sizeof(value));

    op_output.emit(entity, "output");
  };

 private:
  Parameter<std::shared_ptr<Allocator>> allocator_;
  int index_ = 1;
};

class SinkOp : public Operator {
 public:
  HOLOSCAN_OPERATOR_FORWARD_ARGS(SinkOp)

  SinkOp() = default;

  void setup(OperatorSpec& spec) override { spec.input<gxf::Entity>("input"); }

  void compute([[maybe_unused]] InputContext& op_input, OutputContext& op_output,
               [[maybe_unused]] ExecutionContext& context) override {
    auto maybe_entity = op_input.receive<holoscan::gxf::Entity>("input");
    if (!maybe_entity) {
      throw std::runtime_error("Failed to receive input");
    }

    auto& entity = static_cast<nvidia::gxf::Entity&>(maybe_entity.value());

    auto maybe_tensor = entity.get<nvidia::gxf::Tensor>();
    if (!maybe_tensor) {
      throw std::runtime_error("Failed to get tensor");
    }
    auto tensor = maybe_tensor.value();
    auto value = *reinterpret_cast<int*>(tensor->pointer());
    std::cout << "Received value: " << value << std::endl;
  };
};

}  // namespace holoscan::ops

class SlangApp : public holoscan::Application {
 public:
  void compose() override {
    using namespace holoscan;

    auto source = make_operator<ops::SourceOp>("Source");
    auto sink = make_operator<ops::SinkOp>("Sink");

    std::ifstream in_stream("shader.slang");
    if (!in_stream.is_open()) {
      throw std::runtime_error("Failed to open shader file");
    }
    std::stringstream shader_string;
    shader_string << in_stream.rdbuf();

    auto slang = make_operator<ops::SlangShaderOp>(
        "Slang", shader_string.str(), make_condition<CountCondition>(10));

    // Define the workflow
    add_flow(source, slang, {{"output", "input_buffer"}});
    add_flow(slang, sink, {{"output_buffer", "input"}});
  }
};

int main() {
  auto app = holoscan::make_application<SlangApp>();
  app->run();

  return 0;
}
