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

#ifndef TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_ASM_COMPILER_H_
#define TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_ASM_COMPILER_H_

#include <array>
#include <cstdint>
#include <tuple>
#include <vector>

#include "const_init.h"
#include "thread_annotations.h"
#include "flat_hash_map.h"
#include "mutex.h"
#include "span.h"
#include "gpu_asm_opts.h"
#include "kernel.h"
#include "port.h"
#include "stream_executor_pimpl.h"
#include "statusor.h"
#if GOOGLE_CUDA
#include "cuda_driver.h"
#endif  // GOOGLE_CUDA

namespace stream_executor {
namespace gpu {
class GpuContext;
}

// Compiles the given PTX string using ptxas and returns the resulting machine
// code (i.e. a cubin) as a byte array. The generated cubin matches the compute
// capabilities of the device associated with 'device_ordinal'.
//
// 'options' is used to query for the CUDA location in case it is
// customized in a passed flag, and for controlling ptxas optimizations.
tsl::StatusOr<std::vector<uint8_t>> CompileGpuAsm(int device_ordinal,
                                                  const char* ptx_contents,
                                                  GpuAsmOpts options);

// Compiles the given PTX string using ptxas and returns the resulting machine
// code (i.e. a cubin) as a byte array. The generated cubin matches the compute
// capabilities provided by 'cc_major' and 'cc_minor'.
//
// 'options' is used to query for the CUDA location in case it is
// customized in a passed flag, and for controlling ptxas optimizations.
tsl::StatusOr<std::vector<uint8_t>> CompileGpuAsm(int cc_major, int cc_minor,
                                                  const char* ptx_contents,
                                                  GpuAsmOpts options);

// Same as CompileGpuAsm, but caches the result, and returns unowned view of
// the compiled binary.
//
// A copy of the string provided in ptx will be made.
tsl::StatusOr<absl::Span<const uint8_t>> CompileGpuAsmOrGetCached(
    int device_ordinal, const char* ptx, GpuAsmOpts compilation_options);

struct CubinOrPTXImage {
  std::string profile;
  std::vector<uint8_t> bytes;
};

// Bundles the GPU machine code (cubins) and PTX if requested and returns the
// resulting binary (i.e. a fatbin) as a byte array.
tsl::StatusOr<std::vector<uint8_t>> BundleGpuAsm(
    std::vector<CubinOrPTXImage> images, GpuAsmOpts options);

struct HsacoImage {
  std::string gfx_arch;
  std::vector<uint8_t> bytes;
};

// Bundles the GPU machine code (HSA Code Object) and returns the resulting
// binary (i.e. a fatbin) as a byte array.
tsl::StatusOr<std::vector<uint8_t>> BundleGpuAsm(
    std::vector<HsacoImage> images, const std::string rocm_root_dir);

// Links multiple relocatable GPU images (e.g. results of ptxas -c) into a
// single image.
tsl::StatusOr<std::vector<uint8_t>> LinkGpuAsm(
    gpu::GpuContext* context, std::vector<CubinOrPTXImage> images);

tsl::StatusOr<std::vector<uint8_t>> LinkUsingNvlink(
    absl::string_view preferred_cuda_dir, gpu::GpuContext* context,
    std::vector<CubinOrPTXImage> images);

std::string FindCudaExecutable(const std::string& binary_name,
                               const std::string& preferred_cuda_dir);

// Runs tool --version and parses its version string.
tsl::StatusOr<std::array<int64_t, 3>> GetToolVersion(
    absl::string_view tool_path);

// On NVIDIA GPUs, returns the CUDA toolkit version supported by the driver,
tsl::StatusOr<std::array<int64_t, 3>> GetAsmCompilerVersion(
    const std::string& preferred_cuda_dir);

#if GOOGLE_CUDA
// Maintains a cache of pointers to loaded kernels
template <typename... Args>
tsl::StatusOr<std::shared_ptr<TypedKernel<Args...>>> LoadKernelOrGetPtr(
    StreamExecutor* executor, absl::string_view kernel_name,
    absl::string_view ptx, absl::Span<const uint8_t> cubin_data) {
  using KernelPtrCacheKey =
      std::tuple<CUcontext, absl::string_view, absl::string_view>;

  static absl::Mutex kernel_ptr_cache_mutex(absl::kConstInit);
  static auto& kernel_ptr_cache ABSL_GUARDED_BY(kernel_ptr_cache_mutex) =
      *new absl::flat_hash_map<KernelPtrCacheKey,
                               std::shared_ptr<TypedKernel<Args...>>>();
  CUcontext current_context = cuda::CurrentContextOrDie();
  KernelPtrCacheKey kernel_ptr_cache_key{current_context, kernel_name, ptx};
  absl::MutexLock lock(&kernel_ptr_cache_mutex);

  auto it = kernel_ptr_cache.find(kernel_ptr_cache_key);
  if (it == kernel_ptr_cache.end()) {
    TF_ASSIGN_OR_RETURN(
        std::shared_ptr<TypedKernel<Args...>> loaded,
        executor->CreateTypedKernel<Args...>(kernel_name, ptx, cubin_data));
    it =
        kernel_ptr_cache.emplace(kernel_ptr_cache_key, std::move(loaded)).first;
  }

  CHECK(it != kernel_ptr_cache.end());
  return it->second;
}
#endif  // GOOGLE_CUDA

}  // namespace stream_executor

#endif  // TENSORFLOW_COMPILER_XLA_STREAM_EXECUTOR_GPU_ASM_COMPILER_H_
