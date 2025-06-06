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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_ASSERT_PREV_DATASET_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_ASSERT_PREV_DATASET_OP_H_

#include "dataset.h"

namespace tensorflow {
namespace data {
namespace experimental {

class AssertPrevDatasetOp : public UnaryDatasetOpKernel {
 public:
  static constexpr char kDatasetType[] = "AssertPrev";
  static constexpr char kInputDataset[] = "input_dataset";
  static constexpr char kTransformations[] = "transformations";
  static constexpr char kOutputTypes[] = "output_types";
  static constexpr char kOutputShapes[] = "output_shapes";

  explicit AssertPrevDatasetOp(OpKernelConstruction* ctx);

 protected:
  void MakeDataset(OpKernelContext* ctx, DatasetBase* input,
                   DatasetBase** output) override;

 private:
  class Dataset;
  DataTypeVector output_types_;
  std::vector<PartialTensorShape> output_shapes_;
};

}  // namespace experimental
}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_EXPERIMENTAL_ASSERT_PREV_DATASET_OP_H_
