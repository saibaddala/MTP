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
#ifndef TENSORFLOW_CORE_KERNELS_COLLECTIVE_NCCL_H_
#define TENSORFLOW_CORE_KERNELS_COLLECTIVE_NCCL_H_

#include "collective.h"

namespace tensorflow {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

class NcclBase : public CollectiveImplementationInterface {
 public:
  explicit NcclBase(CollectiveType type, const string& name);
  ~NcclBase() override = default;

  // No-op for this collective implementation.
  Status InitializeCollectiveParams(CollectiveParams* col_params) override;

  // Initializes the device objects and device localities.
  Status InitializeCollectiveContext(
      std::shared_ptr<CollectiveContext> col_ctx) override;

 protected:
  const CollectiveType type_;
  const string name_;
  std::shared_ptr<CollectiveContext> col_ctx_;
  const CollectiveParams* col_params_;  // Not owned
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_COLLECTIVE_NCCL_H_
