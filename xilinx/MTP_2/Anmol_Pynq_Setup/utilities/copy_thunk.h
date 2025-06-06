/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_COPY_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_COPY_THUNK_H_

#include "hlo_instruction.h"
#include "buffer_assignment.h"
#include "buffer_allocations.h"
#include "thunk.h"
#include "stream_executor.h"

namespace xla {
namespace gpu {

// A thunk that copies data from a device buffer to another device buffer.
class DeviceToDeviceCopyThunk : public Thunk {
 public:
  // Constructs a CopyThunk that copies host data from `source_buffer` to the
  // device buffer `destination_buffer`. `mem_size` is the size of the data in
  // bytes.
  DeviceToDeviceCopyThunk(ThunkInfo thunk_info,
                          const BufferAllocation::Slice& source_buffer,
                          const BufferAllocation::Slice& destination_buffer,
                          uint64_t mem_size, mlir::Value source_value,
                          mlir::Value destination_value);

  DeviceToDeviceCopyThunk(const DeviceToDeviceCopyThunk&) = delete;
  DeviceToDeviceCopyThunk& operator=(const DeviceToDeviceCopyThunk&) = delete;

  Status ExecuteOnStream(const ExecuteParams& params) override;

  void ClearCompileTimeInfo() override {
    Thunk::ClearCompileTimeInfo();
    source_value_ = nullptr;
    destination_value_ = nullptr;
  }

  const BufferAllocation::Slice& source() const { return source_buffer_; }
  const BufferAllocation::Slice& destination() const {
    return destination_buffer_;
  }
  uint64_t size_bytes() const { return mem_size_; }
  mlir::Value source_value() const { return source_value_; }
  mlir::Value destination_value() const { return destination_value_; }

 private:
  const BufferAllocation::Slice source_buffer_;
  const BufferAllocation::Slice destination_buffer_;
  const uint64_t mem_size_;
  mlir::Value source_value_;
  mlir::Value destination_value_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_COPY_THUNK_H_
