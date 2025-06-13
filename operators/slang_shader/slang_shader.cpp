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

#include "slang_shader.hpp"

#include <cuda_runtime.h>

// Slang includes
#include <slang-com-helper.h>
#include <slang-com-ptr.h>
#include <slang.h>

#include <nlohmann/json.hpp>

#include <holoscan/core/execution_context.hpp>
#include <holoscan/core/fragment.hpp>
#include <holoscan/core/gxf/entity.hpp>
#include <holoscan/core/io_context.hpp>
#include <holoscan/core/resources/gxf/unbounded_allocator.hpp>

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

#include "holoscan_slang.hpp"

namespace {

static const char* get_error_string(Slang::Result result) {
  switch (result) {
    case SLANG_E_NOT_IMPLEMENTED:
      return "SLANG_E_NOT_IMPLEMENTED";
    case SLANG_E_NO_INTERFACE:
      return "SLANG_E_NO_INTERFACE";
    case SLANG_E_ABORT:
      return "SLANG_E_ABORT";
    case SLANG_E_INVALID_HANDLE:
      return "SLANG_E_INVALID_HANDLE";
    case SLANG_E_INVALID_ARG:
      return "SLANG_E_INVALID_ARG";
    case SLANG_E_OUT_OF_MEMORY:
      return "SLANG_E_OUT_OF_MEMORY";
    case SLANG_E_BUFFER_TOO_SMALL:
      return "SLANG_E_BUFFER_TOO_SMALL";
    case SLANG_E_UNINITIALIZED:
      return "SLANG_E_UNINITIALIZED";
    case SLANG_E_PENDING:
      return "SLANG_E_PENDING";
    case SLANG_E_CANNOT_OPEN:
      return "SLANG_E_CANNOT_OPEN";
    case SLANG_E_NOT_FOUND:
      return "SLANG_E_NOT_FOUND";
    case SLANG_E_INTERNAL_FAIL:
      return "SLANG_E_INTERNAL_FAIL";
    case SLANG_E_NOT_AVAILABLE:
      return "SLANG_E_NOT_AVAILABLE";
    case SLANG_E_TIME_OUT:
      return "SLANG_E_TIME_OUT";
    default:
      return "Unknown Slang error";
  }
}

}  // namespace

#define CUDA_CALL(stmt, ...)                                                               \
  ({                                                                                       \
    cudaError_t _holoscan_cuda_err = stmt;                                                 \
    if (cudaSuccess != _holoscan_cuda_err) {                                               \
      throw std::runtime_error(                                                            \
          fmt::format("CUDA Runtime call {} in line {} of file {} failed with '{}' ({}).", \
                      #stmt,                                                               \
                      __LINE__,                                                            \
                      __FILE__,                                                            \
                      cudaGetErrorString(_holoscan_cuda_err),                              \
                      static_cast<int>(_holoscan_cuda_err)));                              \
    }                                                                                      \
  })

/// Deleter for unique_ptr of CUDA objects
template <typename T, cudaError_t func(T)>
struct Deleter {
  typedef T pointer;
  void operator()(T value) const { func(value); }
};
using UniqueCudaLibrary =
    std::unique_ptr<cudaLibrary_t, Deleter<cudaLibrary_t, &cudaLibraryUnload>>;

#define SLANG_CALL(stmt)                                                            \
  ({                                                                                \
    Slang::Result _slang_result = stmt;                                             \
    if (SLANG_FAILED(_slang_result)) {                                              \
      throw std::runtime_error(                                                     \
          fmt::format("Slang call {} in line {} of file {} failed with '{}' ({}).", \
                      #stmt,                                                        \
                      __LINE__,                                                     \
                      __FILE__,                                                     \
                      get_error_string(_slang_result),                              \
                      static_cast<int>(_slang_result)));                            \
    }                                                                               \
    _slang_result;                                                                  \
  })

#define SLANG_DIAGNOSE_IF_NEEDED(diagnostics_blob)                        \
  if (diagnostics_blob) {                                                 \
    HOLOSCAN_LOG_INFO("Slang compilation diagnostics: {}",                \
                      (const char*)diagnostics_blob->getBufferPointer()); \
  }

namespace holoscan::ops {

// Implementation struct containing all Slang-related details
class SlangShaderOp::Impl {
 public:
  Impl(const std::string& shader_source) {
    // First we need to create slang global session with work with the Slang API.
    SLANG_CALL(slang::createGlobalSession(global_session_.writeRef()));

    // Create Session
    slang::SessionDesc sessionDesc;

    // Set up target description for SPIR-V
    slang::TargetDesc targetDesc;
    targetDesc.format = SLANG_PTX;

    sessionDesc.targets = &targetDesc;
    sessionDesc.targetCount = 1;

    // Create session
    SLANG_CALL(global_session_->createSession(sessionDesc, session_.writeRef()));

    // Load module from source string
    Slang::ComPtr<ISlangBlob> diagnostics_blob;

    holoscan_module_ = session_->loadModuleFromSourceString(
        "holoscan", "holoscan.slang", holoscan_slang, diagnostics_blob.writeRef());
    // Check for compilation errors
    SLANG_DIAGNOSE_IF_NEEDED(diagnostics_blob);
    if (!holoscan_module_) {
      throw std::runtime_error("Failed to compile Slang module");
    }

    compile_shader(shader_source);
  }

  void compile_shader(const std::string& shader_source) {
    // Load module from source string
    Slang::ComPtr<ISlangBlob> diagnostics_blob;

    user_module_ = session_->loadModuleFromSourceString(
        "user", "user.slang", shader_source.c_str(), diagnostics_blob.writeRef());
    // Check for compilation errors
    SLANG_DIAGNOSE_IF_NEEDED(diagnostics_blob);
    if (!user_module_) {
      throw std::runtime_error("Failed to compile Slang module");
    }

    // Find entry points in the module
    const int entry_point_count = user_module_->getDefinedEntryPointCount();
    if (entry_point_count == 0) {
      throw std::runtime_error("Warning: No entry points found in shader");
    }

    // Get the first entry point
    Slang::ComPtr<slang::IEntryPoint> entryPoint;
    SLANG_CALL(user_module_->getDefinedEntryPoint(0, entryPoint.writeRef()));

    // Create composite component type (module + entry point)
    std::array<slang::IComponentType*, 2> component_types = {user_module_, entryPoint};
    Slang::ComPtr<slang::IComponentType> composed_program;

    SLANG_CALL(session_->createCompositeComponentType(component_types.data(),
                                                      component_types.size(),
                                                      composed_program.writeRef(),
                                                      diagnostics_blob.writeRef()));
    SLANG_DIAGNOSE_IF_NEEDED(diagnostics_blob);

    // Link the program
    SLANG_CALL(composed_program->link(linked_program_.writeRef(), diagnostics_blob.writeRef()));
    SLANG_DIAGNOSE_IF_NEEDED(diagnostics_blob);

    // Optionally generate target code
    Slang::ComPtr<ISlangBlob> ptx_code;
    SLANG_CALL(linked_program_->getEntryPointCode(0,  // entryPointIndex
                                                  0,  // targetIndex
                                                  ptx_code.writeRef(),
                                                  diagnostics_blob.writeRef()));
    SLANG_DIAGNOSE_IF_NEEDED(diagnostics_blob);

    cuda_library_.reset([&ptx_code] {
      cudaLibrary_t library;
      CUDA_CALL(cudaLibraryLoadData(
          &library, ptx_code->getBufferPointer(), nullptr, nullptr, 0, nullptr, nullptr, 0));
      return library;
    }());
  }

  Slang::ComPtr<slang::IComponentType> linked_program_;

  class ParameterWrapper {
   public:
    template <typename typeT>
    explicit ParameterWrapper(OperatorSpec& spec, Parameter<typeT>* param, const std::string& name)
        : value_(reinterpret_cast<void*>(param),
                 [](void* value) { delete static_cast<Parameter<typeT>*>(value); }) {
      spec.param<typeT>(*param, name.c_str());
    }
    std::shared_ptr<void> value_;
  };

  std::list<std::unique_ptr<ParameterWrapper>> parameters_;

  class CommandWorkspace {
   public:
    CommandWorkspace(InputContext& op_input, OutputContext& op_output, ExecutionContext& context)
        : op_input_(op_input), op_output_(op_output), context_(context) {}
    InputContext& op_input_;
    OutputContext& op_output_;
    ExecutionContext& context_;

    std::map<std::string, gxf::Entity> entities_;
  };

  class Command {
   public:
    virtual ~Command() = default;
    virtual void execute(CommandWorkspace& workspace) = 0;
  };

  class CommandInput : public Command {
   public:
    CommandInput(const std::string& name) : name_(name) {}
    std::string name_;
    void execute(CommandWorkspace& workspace) override {
      std::cout << "CommandInput: " << name_ << std::endl;
      auto maybe_entity = workspace.op_input_.receive<gxf::Entity>(name_.c_str());
      if (!maybe_entity.has_value()) {
        throw std::runtime_error(
            fmt::format("Failed to receive entity from input port {}.", name_));
      }
      workspace.entities_[name_] = maybe_entity.value();
    }
  };

  class CommandOutput : public Command {
   public:
    CommandOutput(const std::string& name) : name_(name) {}
    std::string name_;
    void execute(CommandWorkspace& workspace) override {
      std::cout << "CommandOutput: " << name_ << std::endl;
      workspace.op_output_.emit(workspace.entities_[name_], name_.c_str());
    }
  };

  class CommandSizeOf : public Command {
   public:
    CommandSizeOf(const std::string& name, const std::string& size_of_name,
                  const Parameter<std::shared_ptr<Allocator>>& allocator)
        : name_(name), size_of_name_(size_of_name), allocator_(allocator) {}

    void execute(CommandWorkspace& workspace) override {
      std::cout << "CommandSizeOf: " << name_ << " " << size_of_name_ << std::endl;
      auto reference_entity = workspace.entities_.find(size_of_name_);
      if (reference_entity == workspace.entities_.end()) {
        throw std::runtime_error(
            fmt::format("Attribute 'size_of': input '{}' not found.", size_of_name_));
      }
      auto maybe_reference_tensor =
          static_cast<nvidia::gxf::Entity&>(reference_entity->second).get<nvidia::gxf::Tensor>();
      if (!maybe_reference_tensor) {
        throw std::runtime_error(
            fmt::format("Attribute 'size_of': input '{}' is not a tensor.", size_of_name_));
      }
      auto reference_tensor = maybe_reference_tensor.value();

      auto entity = gxf::Entity::New(&workspace.context_);
      auto tensor = static_cast<nvidia::gxf::Entity&>(entity).add<nvidia::gxf::Tensor>().value();
      // Get Handle to underlying nvidia::gxf::Allocator from std::shared_ptr<holoscan::Allocator>
      auto gxf_allocator = nvidia::gxf::Handle<nvidia::gxf::Allocator>::Create(
          workspace.context_.context(),
          allocator_.has_value() ? allocator_.get()->gxf_cid()
                                 : allocator_.default_value()->gxf_cid());
      tensor->reshape<int>(
          reference_tensor->shape(), nvidia::gxf::MemoryStorageType::kHost, gxf_allocator.value());

      workspace.entities_[name_] = entity;
    }

   private:
    const std::string name_;
    const std::string size_of_name_;
    const Parameter<std::shared_ptr<Allocator>>& allocator_;
  };

  class CommandLaunch : public Command {
   public:
    CommandLaunch(const std::string& name) : name_(name) {}
    std::string name_;
    void execute(CommandWorkspace& workspace) override {
      std::cout << "CommandLaunch: " << name_ << std::endl;
    }
  };

  std::list<std::unique_ptr<Command>> pre_launch_commands_;
  std::list<std::unique_ptr<Command>> launch_commands_;
  std::list<std::unique_ptr<Command>> post_launch_commands_;

  Parameter<std::shared_ptr<Allocator>> allocator_;

 private:
  // Slang compilation components
  Slang::ComPtr<slang::IGlobalSession> global_session_;
  Slang::ComPtr<slang::ISession> session_;
  Slang::ComPtr<slang::IModule> holoscan_module_;
  Slang::ComPtr<slang::IModule> user_module_;

  UniqueCudaLibrary cuda_library_;
};

void SlangShaderOp::init(const std::string& shader_source) {
  impl_ = std::make_shared<Impl>(shader_source);
}

void SlangShaderOp::setup(OperatorSpec& spec) {
  spec.param(impl_->allocator_,
             "allocator",
             "Allocator for output buffers.",
             "Allocator for output buffers.",
             std::static_pointer_cast<Allocator>(
                 fragment()->make_resource<UnboundedAllocator>("allocator")));

  // Get reflection from the linked program and convert to JSON
  Slang::ComPtr<ISlangBlob> reflection_blob;
  SLANG_CALL(impl_->linked_program_->getLayout(0)->toJson(reflection_blob.writeRef()));
  // std::cout << "Reflection: " << (const char*)reflection_blob->getBufferPointer() << std::endl;

  nlohmann::json reflection =
      nlohmann::json::parse((const char*)reflection_blob->getBufferPointer());

  // Get Slang parameters and setup input, outputs and parameters
  for (auto& parameter : reflection["parameters"]) {
    if (!parameter.contains("userAttribs")) {
      continue;
    }

    // Check the user attributes and check for Holoscan attributes
    for (auto& user_attrib : parameter["userAttribs"]) {
      // Ignore non-Holoscan attributes
      const std::string user_attrib_name = user_attrib["name"];
      const std::string holoscan_prefix = "holoscan_";
      if (user_attrib_name.find(holoscan_prefix) != 0) {
        continue;
      }

      const std::string attrib_name = user_attrib_name.substr(holoscan_prefix.size());

      if (attrib_name == "input") {
        // input
        if ((parameter["type"]["kind"] != "resource") ||
            (parameter["type"]["baseShape"] != "structuredBuffer")) {
          throw std::runtime_error(fmt::format(
              "Attribute '{}' supports structured buffers only and cannot be applied to '{}'.",
              attrib_name,
              parameter["name"].get<std::string>()));
        }
        const std::string input_name = user_attrib["arguments"].at(0);
        spec.input<gxf::Entity>(input_name);
        impl_->pre_launch_commands_.push_back(std::make_unique<Impl::CommandInput>(input_name));
      } else if (attrib_name == "output") {
        // output
        if ((parameter["type"]["kind"] != "resource") ||
            (parameter["type"]["baseShape"] != "structuredBuffer")) {
          throw std::runtime_error(fmt::format(
              "Attribute '{}' supports structured buffers only and cannot be applied to '{}'.",
              attrib_name,
              parameter["name"].get<std::string>()));
        }
        const std::string output_name = user_attrib["arguments"].at(0);
        spec.output<gxf::Entity>(output_name);
        impl_->post_launch_commands_.push_back(std::make_unique<Impl::CommandOutput>(output_name));
      } else if (attrib_name == "size_of") {
        // size_of
        if ((parameter["type"]["kind"] != "resource") ||
            (parameter["type"]["baseShape"] != "structuredBuffer")) {
          throw std::runtime_error(fmt::format(
              "Attribute '{}' supports structured buffers only and cannot be applied to '{}'.",
              attrib_name,
              parameter["name"].get<std::string>()));
        }
        const std::string size_of_name = user_attrib["arguments"].at(0);
        impl_->pre_launch_commands_.push_back(std::make_unique<Impl::CommandSizeOf>(
            parameter["name"], size_of_name, impl_->allocator_));
      } else if (attrib_name == "parameter") {
        // parameter
        if ((parameter["binding"]["kind"] != "uniform") ||
            (parameter["type"]["kind"] != "scalar")) {
          throw std::runtime_error(fmt::format(
              "Attribute '{}' supports scalar uniforms only and cannot be applied to '{}'.",
              attrib_name,
              parameter["name"].get<std::string>()));
        }
        const std::string param_name = user_attrib["arguments"].at(0);
        if (parameter["type"]["scalarType"] == "bool") {
          impl_->parameters_.push_back(
              std::make_unique<Impl::ParameterWrapper>(spec, new Parameter<bool>(), param_name));
        } else if (parameter["type"]["scalarType"] == "int32") {
          impl_->parameters_.push_back(
              std::make_unique<Impl::ParameterWrapper>(spec, new Parameter<int32_t>(), param_name));
        } else if (parameter["type"]["scalarType"] == "uint32") {
          impl_->parameters_.push_back(std::make_unique<Impl::ParameterWrapper>(
              spec, new Parameter<uint32_t>(), param_name));
        } else if (parameter["type"]["scalarType"] == "int64") {
          impl_->parameters_.push_back(
              std::make_unique<Impl::ParameterWrapper>(spec, new Parameter<int64_t>(), param_name));
        } else if (parameter["type"]["scalarType"] == "uint64") {
          impl_->parameters_.push_back(std::make_unique<Impl::ParameterWrapper>(
              spec, new Parameter<uint64_t>(), param_name));
        } else if (parameter["type"]["scalarType"] == "float32") {
          impl_->parameters_.push_back(
              std::make_unique<Impl::ParameterWrapper>(spec, new Parameter<float>(), param_name));
        } else if (parameter["type"]["scalarType"] == "float64") {
          impl_->parameters_.push_back(
              std::make_unique<Impl::ParameterWrapper>(spec, new Parameter<double>(), param_name));
        } else if (parameter["type"]["scalarType"] == "int8") {
          impl_->parameters_.push_back(
              std::make_unique<Impl::ParameterWrapper>(spec, new Parameter<int8_t>(), param_name));
        } else if (parameter["type"]["scalarType"] == "uint8") {
          impl_->parameters_.push_back(
              std::make_unique<Impl::ParameterWrapper>(spec, new Parameter<uint8_t>(), param_name));
        } else if (parameter["type"]["scalarType"] == "int16") {
          impl_->parameters_.push_back(
              std::make_unique<Impl::ParameterWrapper>(spec, new Parameter<int16_t>(), param_name));
        } else if (parameter["type"]["scalarType"] == "uint16") {
          impl_->parameters_.push_back(std::make_unique<Impl::ParameterWrapper>(
              spec, new Parameter<uint16_t>(), param_name));
        } else {
          throw std::runtime_error(
              fmt::format("Attribute '{}' unsupported scalar type '{}' for parameter '{}'.",
                          attrib_name,
                          parameter["type"]["scalarType"].get<std::string>(),
                          parameter["name"].get<std::string>()));
        }
      } else if (attrib_name == "size_of") {
      } else if (attrib_name == "grid_size_of") {
      } else {
        throw std::runtime_error("Unknown user attribute: " + user_attrib_name);
      }
    }
  }

  // Get Slang entry points and launch commands
  for (auto& entry_point : reflection["entryPoints"]) {
    const std::string entry_point_name = entry_point["name"];
    impl_->launch_commands_.push_back(std::make_unique<Impl::CommandLaunch>(entry_point_name));
  }

#if 0
  slang::VariableLayoutReflection* global_params_var_layout = program_layout->getGlobalParamsVarLayout();

  auto scopeTypeLayout = global_params_var_layout->getTypeLayout();
  std::cout << "Type layout: " << (int)scopeTypeLayout->getKind() << std::endl;
  switch (scopeTypeLayout->getKind()) {
    case slang::TypeReflection::Kind::Struct: {
      std::cout << "parameters: " << std::endl;

      int paramCount = scopeTypeLayout->getFieldCount();
      for (int i = 0; i < paramCount; i++) {
        std::cout << "- ";

        auto param = scopeTypeLayout->getFieldByIndex(i);
        //printVarLayout(param, &scopeOffsets);
      }
    } break;
    default:
      std::cout << "Unknown type layout" << std::endl;
      break;
  }
#endif
}

void SlangShaderOp::compute(InputContext& op_input, OutputContext& op_output,
                            ExecutionContext& context) {
  Impl::CommandWorkspace workspace(op_input, op_output, context);

  for (auto& command : impl_->pre_launch_commands_) { command->execute(workspace); }

  for (auto& command : impl_->launch_commands_) { command->execute(workspace); }

  for (auto& command : impl_->post_launch_commands_) { command->execute(workspace); }
}
}  // namespace holoscan::ops
