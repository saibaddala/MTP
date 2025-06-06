/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_KERNELS_COLLECTIVE_NCCL_ALL_TO_ALL_H_
#define TENSORFLOW_CORE_KERNELS_COLLECTIVE_NCCL_ALL_TO_ALL_H_

#include "collective_nccl.h"

namespace tensorflow {
#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

class NcclAllToAll : public NcclBase {
 public:
  NcclAllToAll() : NcclBase(ALL_TO_ALL_COLLECTIVE, "NcclAllToAll") {}
  ~NcclAllToAll() override = default;

  // Hands off all-to-all to NcclManager.
  void Run(StatusCallback done) override;
};

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_COLLECTIVE_NCCL_ALL_TO_ALL_H_
