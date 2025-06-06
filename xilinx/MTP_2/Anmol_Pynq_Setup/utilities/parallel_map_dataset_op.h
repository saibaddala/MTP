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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_PARALLEL_MAP_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_PARALLEL_MAP_DATASET_OP_H_

#include "captured_function.h"
#include "dataset_utils.h"
#include "dataset.h"

namespace tensorflow {
namespace data {

class ParallelMapDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "ParallelMap";
  static constexpr const char* const kInputDataset = "input_dataset";
  static constexpr const char* const kOtherArguments = "other_arguments";
  static constexpr const char* const kNumParallelCalls = "num_parallel_calls";
  static constexpr const char* const kFunc = "f";
  static constexpr const char* const kTarguments = "Targuments";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";
  static constexpr const char* const kUseInterOpParallelism =
      "use_inter_op_parallelism";
  static constexpr const char* const kDeterministic = "deterministic";
  static constexpr const char* const kSloppy = "sloppy";
  static constexpr const char* const kPreserveCardinality =
      "preserve_cardinality";

  explicit ParallelMapDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  const int op_version_;
  std::shared_ptr<FunctionMetadata> func_metadata_ = nullptr;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
  bool sloppy_;
  bool preserve_cardinality_;
  DeterminismPolicy deterministic_;

  friend std::unique_ptr<DatasetBase> MakeDataServiceUncompressDataset(
      DatasetBase* input, std::unique_ptr<CapturedFunction> captured_function,
      const DataTypeVector& output_types,
      const std::vector<PartialTensorShape>& output_shapes);
};

// Used by tf.data service to create a map dataset for uncompression.
std::unique_ptr<DatasetBase> MakeDataServiceUncompressDataset(
    DatasetBase* input, std::unique_ptr<CapturedFunction> captured_function,
    const DataTypeVector& output_types,
    const std::vector<PartialTensorShape>& output_shapes);

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_PARALLEL_MAP_DATASET_OP_H_
