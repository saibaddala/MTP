/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_DATA_GET_OPTIONS_OP_H_
#define TENSORFLOW_CORE_KERNELS_DATA_GET_OPTIONS_OP_H_

#include "op_kernel.h"

namespace tensorflow {
namespace data {

// TODO(jsimsa): Provide class-level documentation for this and the other ops.
class GetOptionsOp : public OpKernel {
 public:
  explicit GetOptionsOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) final;

  string TraceString(const OpKernelContext& ctx, bool verbose) const override;
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_DATA_GET_OPTIONS_OP_H_
