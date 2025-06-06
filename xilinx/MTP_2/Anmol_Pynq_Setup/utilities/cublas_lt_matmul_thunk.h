/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUBLAS_LT_MATMUL_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUBLAS_LT_MATMUL_THUNK_H_

#include <memory>
#include <optional>
#include <utility>

#include "buffer_assignment.h"
#include "matmul_utils.h"
#include "thunk.h"
#include "status.h"
#include "statusor.h"
#if GOOGLE_CUDA
#include "cuda_blas_lt.h"
#else
#include "rocm_config.h"
#include "hip_blas_lt.h"
#endif  // GOOGLE_CUDA

namespace xla {
namespace gpu {

class CublasLtMatmulThunk : public Thunk {
 public:
  CublasLtMatmulThunk(ThunkInfo thunk_info, GemmConfig gemm_config,
                      se::gpu::BlasLt::Epilogue epilogue, int64_t algorithm_idx,
                      BufferAllocation::Slice a_buffer,
                      BufferAllocation::Slice b_buffer,
                      BufferAllocation::Slice c_buffer,
                      BufferAllocation::Slice d_buffer,
                      BufferAllocation::Slice bias_buffer /* may be null */,
                      BufferAllocation::Slice aux_buffer /* may be null */,
                      BufferAllocation::Slice a_scale_buffer /* may be null */,
                      BufferAllocation::Slice b_scale_buffer /* may be null */,
                      BufferAllocation::Slice c_scale_buffer /* may be null */,
                      BufferAllocation::Slice d_scale_buffer /* may be null */,
                      BufferAllocation::Slice d_amax_buffer /* may be null */);

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  StatusOr<cublas_lt::MatmulPlan*> GetMatmulPlan(
      const stream_executor::Stream* stream);

  absl::Mutex matmul_plans_cache_mutex_;
  absl::flat_hash_map<const stream_executor::Stream*,
                      std::unique_ptr<cublas_lt::MatmulPlan>>
      matmul_plans_cache_ ABSL_GUARDED_BY(matmul_plans_cache_mutex_);

  GemmConfig gemm_config_;
  se::gpu::BlasLt::Epilogue epilogue_;
  int64_t algorithm_idx_;
  BufferAllocation::Slice a_buffer_;
  BufferAllocation::Slice b_buffer_;
  BufferAllocation::Slice c_buffer_;
  BufferAllocation::Slice d_buffer_;
  BufferAllocation::Slice bias_buffer_;
  BufferAllocation::Slice aux_buffer_;
  BufferAllocation::Slice a_scale_buffer_;
  BufferAllocation::Slice b_scale_buffer_;
  BufferAllocation::Slice c_scale_buffer_;
  BufferAllocation::Slice d_scale_buffer_;
  BufferAllocation::Slice d_amax_buffer_;
  std::optional<se::gpu::BlasLt::MatmulAlgorithm> algorithm_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_CUBLAS_LT_MATMUL_THUNK_H_
