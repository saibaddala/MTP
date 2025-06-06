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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_INFEED_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_INFEED_THUNK_H_

#include "hlo_instruction.h"
#include "buffer_assignment.h"
#include "buffer_allocations.h"
#include "thunk.h"
#include "stream_executor.h"

namespace xla {
namespace gpu {

// A thunk that infeeds data. Data must be already resident on the
// device. This thunk performs an intra-device copy from that location
// to the buffer allocated for the infeed op.
class InfeedThunk : public Thunk {
 public:
  // Constructs a InfeedThunk that copies data from the on-device
  // infeed queue into the buffers in the given shape tree.
  InfeedThunk(ThunkInfo thunk_info, std::vector<ShapedSlice> dest_slices);

  InfeedThunk(const InfeedThunk&) = delete;
  InfeedThunk& operator=(const InfeedThunk&) = delete;

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  const std::vector<ShapedSlice> dest_slices_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_INFEED_THUNK_H_
