/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_GPU_OBJECT_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_GPU_OBJECT_H_

#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "opencl_wrapper.h"
#include "access_type.h"
#include "data_type.h"
#include "status.h"
#include "gpu_object_desc.h"

namespace tflite {
namespace gpu {
namespace cl {

struct GPUResourcesWithValue {
  GenericGPUResourcesWithValue generic;

  std::vector<std::pair<std::string, cl_mem>> buffers;
  std::vector<std::pair<std::string, cl_mem>> images2d;
  std::vector<std::pair<std::string, cl_mem>> image2d_arrays;
  std::vector<std::pair<std::string, cl_mem>> images3d;
  std::vector<std::pair<std::string, cl_mem>> image_buffers;
  std::vector<std::pair<std::string, cl_mem>> custom_memories;

  void AddFloat(const std::string& name, float value) {
    generic.AddFloat(name, value);
  }
  void AddInt(const std::string& name, int value) {
    generic.AddInt(name, value);
  }
};

class GPUObject {
 public:
  GPUObject() = default;
  // Move only
  GPUObject(GPUObject&& obj_desc) = default;
  GPUObject& operator=(GPUObject&& obj_desc) = default;
  GPUObject(const GPUObject&) = delete;
  GPUObject& operator=(const GPUObject&) = delete;
  virtual ~GPUObject() = default;
  virtual absl::Status GetGPUResources(
      const GPUObjectDescriptor* obj_ptr,
      GPUResourcesWithValue* resources) const = 0;
};

using GPUObjectPtr = std::unique_ptr<GPUObject>;

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_GPU_OBJECT_H_
