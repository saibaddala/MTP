/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_GL_RUNTIME_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_GL_RUNTIME_H_

#include <functional>
#include <vector>

#include "gpu_info.h"
#include "status.h"
#include "types.h"
#include "command_queue.h"
#include "gl_buffer.h"
#include "gl_program.h"
#include "gl_shader.h"
#include "object.h"
#include "object_manager.h"
#include "shared_buffer.h"
#include "runtime_options.h"
#include "stats.h"
#include "variable.h"

namespace tflite {
namespace gpu {
namespace gl {

// Runtime compiles code and executes it once all code is compiled. It creates
// intermediate objects and destroys them when runtime is destroyed.
class Runtime {
 public:
  Runtime(const RuntimeOptions& options, const GpuInfo& gpu_info,
          CommandQueue* command_queue, const ObjectManager* external_objects);

  // Takes parameters and objects and prepares GL program.
  absl::Status AddProgram(const GlShader& shader,
                          const std::vector<Variable>& parameters,
                          const std::vector<Object>& objects,
                          const uint3& num_workgroups);

  // Needs to be called once all programs and shaders has been added to runtime.
  absl::Status PrepareForExecution();

  // Executes all compiled programs.
  // TODO(akulik): add more controls over execution. Execution policy?
  absl::Status Execute();

  // Gets access to objects created while executing generated code.
  const ObjectManager* internal_objects() const { return &internal_objects_; }

  CommandQueue* command_queue() { return command_queue_; }

  RuntimeStats stats() const {
    RuntimeStats stats;
    stats.const_objects = const_objects_.stats();
    stats.internal_objects = internal_objects_.stats();
    if (external_objects_) {
      stats.external_objects = external_objects_->stats();
    }
    return stats;
  }

 private:
  absl::Status AllocateInternalObject(const Object& object);

  absl::Status AllocateConstObject(const Object& object, uint32_t* id);

  // Goes over objects in programs and decides how to allocate them to
  // minimize total allocated memory. Returns a collection of objects to be
  // allocated and shared by internal objects.
  absl::Status AssignInternalObjects(std::vector<Object>* objects);

  const RuntimeOptions options_;
  const GpuInfo gpu_info_;
  const ObjectManager* external_objects_;
  CommandQueue* command_queue_;

  ObjectManager internal_objects_;
  ObjectManager const_objects_;
  uint32_t next_const_id_ = 0;  // id for const objects

  std::unique_ptr<SharedBufferData> shared_readonly_buffer_;

  using BindFunc = std::function<absl::Status()>;

  // Encapsulates a program and all object to bind before dispatch.
  struct CompiledProgramDescriptor {
    GlProgram program;
    uint3 num_workgroups;

    std::vector<BindFunc> bindings;
    std::vector<Object> refs;
  };

  std::vector<CompiledProgramDescriptor> programs_;
};

}  // namespace gl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_GL_RUNTIME_H_
