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

#ifndef TENSORFLOW_CORE_KERNELS_MKL_MKL_KERNEL_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_MKL_MKL_KERNEL_UTIL_H_

#ifdef INTEL_MKL

#include "dnnl.hpp"
#include "testlib.h"
#include "session.h"
#include "status.h"

using dnnl::memory;

namespace tensorflow {

class MklTestingUtil {
 public:
  static void RunMklQuantizeOp(const Tensor& input, const float input_min,
                               const float input_max, DataType type,
                               string mode, Tensor* output);
  static void RunDequantizeOp(const Tensor& input, const Tensor& input_min,
                              const Tensor& input_max, string mode,
                              Tensor* output);

  static void RunGraph(const tensorflow::GraphDef graph_def,
                       const string& fetch, Tensor* output);
  template <typename T>
  static void ComputeMinMax(const Tensor& tf_tensor, T* tensor_min,
                            T* tensor_max) {
    auto eigen_tensor = tf_tensor.flat<T>();
    Eigen::Tensor<T, 0, Eigen::RowMajor> min = eigen_tensor.minimum();
    Eigen::Tensor<T, 0, Eigen::RowMajor> max = eigen_tensor.maximum();
    *tensor_min = min();
    *tensor_max = max();
  }
};

#ifdef ENABLE_ONEDNN_V3
// Since oneDNN v3.x exposes only an opaque memory descriptor, it is no longer
// possible to cache the entire filter memory descriptor as is. So we store
// all relevant information about it in the following class.
//
// TODO(intel-tf): When oneDNN major version changes to v4.x, weight
// caching may not work as expected if the underlying memory descriptor
// has changed (i.e. compared to v3.x). We have to return a status here
// to catch oneDNN major version change to avoid unexpected results.
class FilterMemoryDesc {
 public:
  FilterMemoryDesc() {}

  explicit FilterMemoryDesc(int ndims, int inner_nblks,
                            memory::data_type data_type,
                            const memory::dims& dims,
                            const memory::dims& inner_blks,
                            const memory::dims& inner_idxs,
                            const memory::dims& strides)
      : ndims_(ndims),
        inner_nblks_(inner_nblks),
        data_type_(data_type),
        dims_(dims),
        inner_blks_(inner_blks),
        inner_idxs_(inner_idxs),
        strides_(strides) {}

  ~FilterMemoryDesc() {}

  bool operator==(const FilterMemoryDesc& other) const {
    return (ndims_ == other.ndims_ && inner_nblks_ == other.inner_nblks_ &&
            data_type_ == other.data_type_ && dims_ == other.dims_ &&
            inner_blks_ == other.inner_blks_ &&
            inner_idxs_ == other.inner_idxs_ && strides_ == other.strides_);
  }

 private:
  int ndims_;
  int inner_nblks_;
  memory::data_type data_type_;
  memory::dims dims_;
  memory::dims inner_blks_;
  memory::dims inner_idxs_;
  memory::dims strides_;
};
#endif  // ENABLE_ONEDNN_V3
}  // namespace tensorflow

#endif  // INTEL_MKL
#endif  // TENSORFLOW_CORE_KERNELS_MKL_MKL_KERNEL_UTIL_H_
