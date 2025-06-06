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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_TESTING_UTIL_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_TESTING_UTIL_H_

#include <memory>
#include <utility>
#include <vector>

#include "data_type.h"
#include "gpu_info.h"
#include "gpu_model.h"
#include "precision.h"
#include "shape.h"
#include "gpu_operation.h"
#include "tensor_desc.h"
#include "tensor.h"

namespace tflite {
namespace gpu {
using TensorInt32 = Tensor<BHWC, DataType::INT32>;
class TestExecutionEnvironment {
 public:
  TestExecutionEnvironment() = default;
  virtual ~TestExecutionEnvironment() = default;

  virtual std::vector<CalculationsPrecision> GetSupportedPrecisions() const = 0;
  virtual std::vector<TensorStorageType> GetSupportedStorages(
      DataType data_type) const = 0;

  virtual const GpuInfo& GetGpuInfo() const = 0;

  absl::Status ExecuteGPUOperation(
      const std::vector<TensorDescriptor*>& src_cpu,
      const std::vector<TensorDescriptor*>& dst_cpu,
      std::unique_ptr<GPUOperation>&& operation);

  template <typename DstTensorType>
  absl::Status ExecuteGpuModel(const std::vector<TensorFloat32>& src_cpu,
                               const std::vector<DstTensorType*>& dst_cpu,
                               GpuModel* gpu_model);

  template <typename DstTensorType>
  absl::Status ExecuteGPUOperation(const std::vector<TensorFloat32>& src_cpu,
                                   std::unique_ptr<GPUOperation>&& operation,
                                   const std::vector<BHWC>& dst_sizes,
                                   const std::vector<DstTensorType*>& dst_cpu);

  absl::Status ExecuteGPUOperation(
      const std::vector<TensorFloat32>& src_cpu,
      std::unique_ptr<GPUOperation>&& operation,
      const std::vector<BHWC>& dst_sizes,
      const std::initializer_list<TensorFloat32*>& dst_cpu) {
    return ExecuteGPUOperation(src_cpu, std::move(operation), dst_sizes,
                               std::vector<TensorFloat32*>(dst_cpu));
  }

  absl::Status ExecuteGPUOperation(
      const std::vector<Tensor5DFloat32>& src_cpu,
      std::unique_ptr<GPUOperation>&& operation,
      const std::vector<BHWDC>& dst_sizes,
      const std::vector<Tensor5DFloat32*>& dst_cpu);

  absl::Status ExecuteGPUOperation(const TensorFloat32& src_cpu,
                                   std::unique_ptr<GPUOperation>&& operation,
                                   const BHWC& dst_size,
                                   TensorFloat32* result) {
    return ExecuteGPUOperation(std::vector<TensorFloat32>{src_cpu},
                               std::move(operation), dst_size, result);
  }

  absl::Status ExecuteGPUOperation(const Tensor5DFloat32& src_cpu,
                                   std::unique_ptr<GPUOperation>&& operation,
                                   const BHWDC& dst_size,
                                   Tensor5DFloat32* result) {
    return ExecuteGPUOperation(std::vector<Tensor5DFloat32>{src_cpu},
                               std::move(operation), dst_size, result);
  }

  absl::Status ExecuteGPUOperation(const std::vector<TensorFloat32>& src_cpu,
                                   std::unique_ptr<GPUOperation>&& operation,
                                   const BHWC& dst_size,
                                   TensorFloat32* result) {
    return ExecuteGPUOperation(
        std::vector<TensorFloat32>{src_cpu}, std::move(operation),
        std::vector<BHWC>{dst_size}, std::vector<TensorFloat32*>{result});
  }

  absl::Status ExecuteGPUOperation(const std::vector<Tensor5DFloat32>& src_cpu,
                                   std::unique_ptr<GPUOperation>&& operation,
                                   const BHWDC& dst_size,
                                   Tensor5DFloat32* result) {
    return ExecuteGPUOperation(
        std::vector<Tensor5DFloat32>{src_cpu}, std::move(operation),
        std::vector<BHWDC>{dst_size}, std::vector<Tensor5DFloat32*>{result});
  }

 protected:
  virtual absl::Status ExecuteGpuOperationInternal(
      const std::vector<TensorDescriptor*>& src_cpu,
      const std::vector<TensorDescriptor*>& dst_cpu,
      std::unique_ptr<GPUOperation>&& operation) = 0;
};

absl::Status PointWiseNear(const std::vector<float>& ref,
                           const std::vector<float>& to_compare,
                           float eps = 0.0f);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASK_TESTING_UTIL_H_
