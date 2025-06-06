/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_CL_RECORDABLE_QUEUE_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_CL_RECORDABLE_QUEUE_H_

#include "cl_command_queue.h"
#include "cl_context.h"
#include "cl_device.h"
#include "cl_operation.h"
#include "opencl_wrapper.h"
#include "status.h"

namespace tflite {
namespace gpu {
namespace cl {

class RecordableQueue {
 public:
  RecordableQueue() = default;
  virtual ~RecordableQueue() = default;

  // Move only
  RecordableQueue(RecordableQueue&& storage) = default;
  RecordableQueue& operator=(RecordableQueue&& storage) = default;
  RecordableQueue(const RecordableQueue&) = delete;
  RecordableQueue& operator=(const RecordableQueue&) = delete;

  virtual bool IsSupported() const { return false; }
  virtual absl::Status Execute(CLCommandQueue* queue) const {
    return absl::UnimplementedError("");
  }
};

}  // namespace cl
}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_CL_RECORDABLE_QUEUE_H_
