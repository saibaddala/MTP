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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_KERNEL_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_KERNEL_THUNK_H_

#include <memory>
#include <string>
#include <vector>

#include "flat_hash_map.h"
#include "span.h"
#include "Operation.h"  // from @llvm-project
#include "Value.h"  // from @llvm-project
#include "hlo_instruction.h"
#include "buffer_assignment.h"
#include "buffer_allocations.h"
#include "kernel_arguments.h"
#include "launch_dimensions.h"
#include "thunk.h"
#include "stream_executor.h"
#include "types.h"

namespace xla {
namespace gpu {

class GpuExecutable;

// This class stores everything that StreamExecutor needs for launching a
// kernel. It implements the ExecuteOnStream interface for GpuExecutable to
// invoke the corresponding kernel.
//
// This is thread-compatible.
class KernelThunk : public Thunk {
 public:
  // Constructs a thunk for the given kernel.
  //
  // KernelThunk takes args as `BufferAllocation::Slice`s (wrapped in
  // `KernelArgument`s). Each slice directly corresponds to an argument or
  // output of the computation. Also, the values must correspond to each arg
  // directly, not to their base allocation (e.g. they can be the result of an
  // `mlir::memref::ViewOp`).
  KernelThunk(mlir::Operation* op, std::string kernel_name,
              absl::Span<const KernelArgument> kernel_arguments,
              LaunchDimensions launch_dimensions);
  KernelThunk(const KernelThunk&) = delete;
  KernelThunk& operator=(const KernelThunk&) = delete;
  ~KernelThunk() override = default;

  std::string ToStringExtra(int indent) const override;

  Status Initialize(const GpuExecutable& executable,
                    se::StreamExecutor* executor) override;
  Status ExecuteOnStream(const ExecuteParams& params) override;

  void ClearCompileTimeInfo() override {
    Thunk::ClearCompileTimeInfo();
    for (auto& value : values_) {
      value = nullptr;
    }
  }

  const std::vector<BufferAllocation::Slice>& arguments() const {
    return args_;
  }
  const std::vector<bool>& written() const { return written_; }

  const std::string& kernel_name() const { return kernel_name_; }
  const LaunchDimensions& launch_dimensions() const {
    return launch_dimensions_;
  }
  absl::Span<const mlir::Value> values() const { return values_; }

 private:
  // Buffer slices passed to the kernel as arguments.
  std::vector<BufferAllocation::Slice> args_;

  // args_[i] is written iff (written_[i] == true).
  std::vector<bool> written_;

  // Entry kernel name for the computation.
  const std::string kernel_name_;

  // The thread and block dimension used to launch the kernel.
  const LaunchDimensions launch_dimensions_;

  // mlir::Value(s) corresponding to the buffer slice arguments.
  std::vector<mlir::Value> values_;

  mutable absl::Mutex mutex_;

  // Loaded kernels for each `StreamExecutor`.  Requires pointer stability of
  // values.
  absl::flat_hash_map<se::StreamExecutor*, std::unique_ptr<se::KernelBase>>
      kernel_cache_ ABSL_GUARDED_BY(mutex_);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_KERNEL_THUNK_H_
