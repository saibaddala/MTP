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

#ifndef TENSORFLOW_CORE_KERNELS_IMAGE_RESIZE_NEAREST_NEIGHBOR_OP_H_
#define TENSORFLOW_CORE_KERNELS_IMAGE_RESIZE_NEAREST_NEIGHBOR_OP_H_

#include "tensor_types.h"
#include "types.h"

namespace tensorflow {
namespace functor {

template <typename Device, typename T, bool half_pixel_centers,
          bool align_corners>
struct ResizeNearestNeighbor {
  bool operator()(const Device& d, typename TTypes<T, 4>::ConstTensor input,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output);
};

template <typename Device, typename T, bool half_pixel_centers,
          bool align_corners>
struct ResizeNearestNeighborGrad {
  bool operator()(const Device& d,
                  typename TTypes<T, 4>::ConstTensor input_grad,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output_grad);
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_IMAGE_RESIZE_NEAREST_NEIGHBOR_OP_H_
