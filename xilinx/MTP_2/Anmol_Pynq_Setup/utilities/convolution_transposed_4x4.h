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

#ifndef TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_4X4_H_
#define TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_4X4_H_

#include <string>
#include <vector>

#include "data_type.h"
#include "operations.h"
#include "shape.h"
#include "status.h"
#include "buffer_desc.h"
#include "gpu_operation.h"
#include "tensor_desc.h"
#include "weights_conversion.h"
#include "weights_layout.h"
#include "tensor.h"
#include "types.h"

namespace tflite {
namespace gpu {

class ConvolutionTransposed4x4 : public GPUOperation {
 public:
  ConvolutionTransposed4x4() = default;
  void GetPossibleKernelWorkGroups(
      TuningType tuning_type, const GpuInfo& gpu_info,
      const KernelInfo& kernel_info,
      std::vector<int3>* work_groups) const override {
    work_groups->push_back(work_group_size_);
  }
  absl::Status BindArguments(ArgumentsBinder* args) override;
  int3 GetGridSize() const override;

  // Move only
  ConvolutionTransposed4x4(ConvolutionTransposed4x4&& operation) = default;
  ConvolutionTransposed4x4& operator=(ConvolutionTransposed4x4&& operation) =
      default;
  ConvolutionTransposed4x4(const ConvolutionTransposed4x4&) = delete;
  ConvolutionTransposed4x4& operator=(const ConvolutionTransposed4x4&) = delete;

  WeightsDescription GetWeightsDescription() const {
    WeightsDescription desc;
    desc.type = DeduceDataTypeFromPrecision(definition_.precision);
    desc.layout = weights_layout_;
    desc.spatial_remap = GetSpatialWeightsRemap();
    return desc;
  }

  enum class WeightsUploadType {
    LOCAL_MEM_ASYNC,
    LOCAL_MEM_BY_THREADS,
    GLOBAL_MEM,
    CONSTANT_MEM,
  };

 private:
  ConvolutionTransposed4x4(const OperationDef& definition,
                           const GpuInfo& gpu_info);

  friend ConvolutionTransposed4x4 CreateConvolutionTransposed4x4(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);
  friend ConvolutionTransposed4x4 CreateConvolutionTransposed4x4DynamicWeights(
      const GpuInfo& gpu_info, const OperationDef& definition,
      const ConvolutionTransposedAttributes& attr);

  void UploadWeights(
      const tflite::gpu::Tensor<OHWI, DataType::FLOAT32>& weights,
      WeightsUploadType weights_upload_type);

  std::vector<int> GetSpatialWeightsRemap() const;

  std::string GenerateConvolutionTransposedCode(
      const GpuInfo& gpu_info, const OperationDef& op_def,
      WeightsUploadType weights_upload_type);

  WeightsLayout weights_layout_;
};

bool IsConvolutionTransposed4x4Supported(
    const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

ConvolutionTransposed4x4 CreateConvolutionTransposed4x4(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

ConvolutionTransposed4x4 CreateConvolutionTransposed4x4DynamicWeights(
    const GpuInfo& gpu_info, const OperationDef& definition,
    const ConvolutionTransposedAttributes& attr);

}  // namespace gpu
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_GPU_COMMON_TASKS_CONVOLUTION_TRANSPOSED_4X4_H_
