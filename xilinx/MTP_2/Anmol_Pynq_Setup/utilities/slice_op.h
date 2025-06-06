/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_KERNELS_SLICE_OP_H_
#define TENSORFLOW_CORE_KERNELS_SLICE_OP_H_

// Functor definition for SliceOp, must be compilable by nvcc.

#include "Tensor"
#include "tensor_types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T, int NDIMS>
struct Slice {
  void operator()(const Device& d, typename TTypes<T, NDIMS>::Tensor output,
                  typename TTypes<T, NDIMS>::ConstTensor input,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& slice_indices,
                  const Eigen::DSizes<Eigen::DenseIndex, NDIMS>& slice_sizes) {
    MaybeWith32BitIndexing<Device>(
        [&](auto output32, auto input32, auto slice_indices32,
            auto slice_sizes32) {
          output32.device(d) = input32.slice(slice_indices32, slice_sizes32);
        },
        output, input, slice_indices, slice_sizes);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_SLICE_OP_H_
