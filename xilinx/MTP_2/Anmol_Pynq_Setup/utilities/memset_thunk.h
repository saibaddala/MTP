/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MEMSET_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MEMSET_THUNK_H_

#include "hlo_instruction.h"
#include "buffer_assignment.h"
#include "thunk.h"
#include "status.h"
#include "stream_executor.h"

// This file contains thunks that set a buffer's elements to a particular value.
// This can be faster than emitting a kernel to set the elements.

namespace xla {
namespace gpu {

// Thunk that zeroes out a given chunk of memory.
class MemzeroThunk : public Thunk {
 public:
  explicit MemzeroThunk(ThunkInfo thunk_info,
                        const BufferAllocation::Slice& dest,
                        mlir::Value dest_value)
      : Thunk(Kind::kMemzero, thunk_info),
        dest_(dest),
        dest_value_(dest_value) {}

  Status ExecuteOnStream(const ExecuteParams& params) override;

  void ClearCompileTimeInfo() override {
    Thunk::ClearCompileTimeInfo();
    dest_value_ = nullptr;
  }

  const BufferAllocation::Slice& destination() const { return dest_; }
  mlir::Value dest_value() const { return dest_value_; }

 private:
  const BufferAllocation::Slice dest_;
  mlir::Value dest_value_;
};

// Thunk that sets a given chunk of memory to a particular 32-bit value.  The
// destination chunk must have size divisible by 32 bits.
class Memset32BitValueThunk : public Thunk {
 public:
  explicit Memset32BitValueThunk(ThunkInfo thunk_info, uint32_t value,
                                 const BufferAllocation::Slice& dest,
                                 mlir::Value dest_value)
      : Thunk(Kind::kMemset32BitValue, thunk_info),
        value_(value),
        dest_(dest),
        dest_value_(dest_value) {}

  Status ExecuteOnStream(const ExecuteParams& params) override;

  void ClearCompileTimeInfo() override {
    Thunk::ClearCompileTimeInfo();
    dest_value_ = nullptr;
  }

  const BufferAllocation::Slice& destination() const { return dest_; }
  uint32_t value() const { return value_; }
  mlir::Value dest_value() const { return dest_value_; }

 private:
  const uint32_t value_;
  const BufferAllocation::Slice dest_;
  mlir::Value dest_value_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_MEMSET_THUNK_H_
