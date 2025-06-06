/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_UTIL_H_

#include <string>

#include "device.h"
#include "device_mgr.h"
#include "collective.h"
#include "device_attributes.pb.h"
#include "tensor.h"
#include "status.h"

namespace tensorflow {
namespace collective_util {

Status InitializeDeviceAndLocality(const DeviceMgr* dev_mgr,
                                   const string& device_name, Device** device,
                                   DeviceLocality* device_locality);
string SubdivPermDebugString(const CollectiveParams& col_params);

// Used for executing a sub-operation, e.g. a merge_op instance, with
// an OpKernelContext based on the one passed into this Op.
class SubContext {
 public:
  OpKernelContext::Params sub_params_;
  gtl::InlinedVector<TensorValue, 4> sub_inputs_;
  gtl::InlinedVector<AllocatorAttributes, 4> sub_input_attr_;
  gtl::InlinedVector<DeviceContext*, 4> sub_input_dc_;
  // Used only for Binary and Unary Ops for which we require
  // the calculation to be in-place on the first input.
  int forward_from_ = 0;
  std::unique_ptr<OpKernelContext> sub_ctx_;
  SubContext(OpKernelContext* ctx, OpKernelContext::Params* params,
             OpKernel* op, Tensor* output, Tensor* input);
  ~SubContext() = default;
};

Status ComputeBinOp(OpKernelContext* op_ctx, OpKernelContext::Params* params,
                    Device* device, OpKernel* op, Tensor* output,
                    Tensor* input);

}  // namespace collective_util
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_COLLECTIVE_UTIL_H_
