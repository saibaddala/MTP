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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_SHARD_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_SHARD_DATASET_OP_H_

#include "dataset.h"

namespace tensorflow {
namespace data {

class ShardDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr const char* const kDatasetType = "Shard";
  static constexpr const char* const kInputDataset = "input_dataset";
  static constexpr const char* const kNumShards = "num_shards";
  static constexpr const char* const kIndex = "index";
  static constexpr const char* const kRequireNonEmpty = "require_non_empty";
  static constexpr const char* const kOutputTypes = "output_types";
  static constexpr const char* const kOutputShapes = "output_shapes";

  explicit ShardDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  bool require_non_empty_;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_SHARD_DATASET_OP_H_
