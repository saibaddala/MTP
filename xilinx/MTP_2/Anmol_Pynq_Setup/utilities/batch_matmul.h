/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_BATCH_MATMUL_H_
#define TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_BATCH_MATMUL_H_

#include "common.h"
#include "cpu_backend_gemm.h"
#include "cpu_backend_gemm_params.h"
#include "common.h"
#include "tensor_utils.h"
#include "types.h"

namespace tflite {
namespace optimized_ops {

inline void BatchMatMul(const RuntimeShape& lhs_shape, const float* lhs_data,
                        const RuntimeShape& rhs_shape, const float* rhs_data,
                        const RuntimeShape& output_shape, float* output_data,
                        CpuBackendContext* context,
                        bool transpose_lhs = false) {
  using ::tflite::cpu_backend_gemm::Gemm;
  using ::tflite::cpu_backend_gemm::GemmParams;
  using ::tflite::cpu_backend_gemm::MatrixParams;
  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(5, rhs_shape);

  // Determine which dimension is the broadcast dimension.
  auto broadcast_dim = [](int lhs_dim, int rhs_dim) {
    if (lhs_dim == rhs_dim) return lhs_dim;
    if (lhs_dim == 1) return rhs_dim;
    TFLITE_DCHECK_EQ(rhs_dim, 1);
    return lhs_dim;
  };

  // Compute the "extent" for iterating on this dimension.
  // If we are broadcasting, then don't advance (i.e return 0).
  auto extent = [](const RuntimeShape& shape, int x) {
    if (shape.Dims(x) == 1) {
      return 0;
    }
    int prod = 1;
    for (int i = x + 1; i < shape.DimensionsCount(); ++i) {
      prod *= shape.Dims(i);
    }
    return prod;
  };

  const int batch_dim0 =
      broadcast_dim(extended_lhs_shape.Dims(0), extended_rhs_shape.Dims(0));
  const int batch_dim1 =
      broadcast_dim(extended_lhs_shape.Dims(1), extended_rhs_shape.Dims(1));
  const int batch_dim2 =
      broadcast_dim(extended_lhs_shape.Dims(2), extended_rhs_shape.Dims(2));

  const int lhs_ext0 = extent(extended_lhs_shape, 0);
  const int lhs_ext1 = extent(extended_lhs_shape, 1);
  const int lhs_ext2 = extent(extended_lhs_shape, 2);
  const int rhs_ext0 = extent(extended_rhs_shape, 0);
  const int rhs_ext1 = extent(extended_rhs_shape, 1);
  const int rhs_ext2 = extent(extended_rhs_shape, 2);

  // Set params for each matrix multiply.
  const int lhs_rows = extended_lhs_shape.Dims(3);
  const int rhs_cols = extended_rhs_shape.Dims(4);
  const int accum_depth = extended_lhs_shape.Dims(4);

  MatrixParams<float> lhs_params;
  if (transpose_lhs) {
    lhs_params.order = cpu_backend_gemm::Order::kColMajor;
  } else {
    lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  }
  lhs_params.rows = lhs_rows;
  lhs_params.cols = accum_depth;

  MatrixParams<float> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = accum_depth;
  rhs_params.cols = rhs_cols;

  MatrixParams<float> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = lhs_rows;
  dst_params.cols = rhs_cols;

  for (int b0 = 0; b0 < batch_dim0; ++b0) {
    const float* lhs_ptr0 = lhs_data + (b0 * lhs_ext0);
    const float* rhs_ptr0 = rhs_data + (b0 * rhs_ext0);
    for (int b1 = 0; b1 < batch_dim1; ++b1) {
      const float* lhs_ptr1 = lhs_ptr0 + b1 * lhs_ext1;
      const float* rhs_ptr1 = rhs_ptr0 + b1 * rhs_ext1;
      for (int b2 = 0; b2 < batch_dim2; ++b2) {
        const float* lhs_ptr2 = lhs_ptr1 + b2 * lhs_ext2;
        const float* rhs_ptr2 = rhs_ptr1 + b2 * rhs_ext2;
        float* out_ptr = output_data + ((b0 * batch_dim1 * batch_dim2) +
                                        b1 * batch_dim2 + b2) *
                                           lhs_rows * rhs_cols;
        GemmParams<float, float> gemm_params;
        cpu_backend_gemm::Gemm(lhs_params, lhs_ptr2, rhs_params, rhs_ptr2,
                               dst_params, out_ptr, gemm_params, context);
      }
    }
  }
}

inline void BatchMatMul(const RuntimeShape& lhs_shape, const int8_t* lhs_data,
                        const RuntimeShape& rhs_shape, const int8_t* rhs_data,
                        const float* scaling_factors,
                        const int32_t* input_offset, int32_t* row_sums,
                        const RuntimeShape& output_shape,
                        int32_t* accum_scratch, float* output_data,
                        bool* compute_row_sums, CpuBackendContext* context) {
  using ::tflite::cpu_backend_gemm::Gemm;
  using ::tflite::cpu_backend_gemm::GemmParams;
  using ::tflite::cpu_backend_gemm::MatrixParams;

  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(5, rhs_shape);

  // Determine which dimension is the broadcast dimension.
  auto broadcast_dim = [](int lhs_dim, int rhs_dim) {
    if (lhs_dim == rhs_dim) return lhs_dim;
    if (lhs_dim == 1) return rhs_dim;
    TFLITE_DCHECK_EQ(rhs_dim, 1);
    return lhs_dim;
  };

  // Compute the "extent" for iterating on this dimension.
  // If we are broadcasting, then don't advance (i.e return 0).
  auto extent = [](const RuntimeShape& shape, int x) {
    if (shape.Dims(x) == 1) {
      return 0;
    }
    int prod = 1;
    for (int i = x + 1; i < shape.DimensionsCount(); ++i) {
      prod *= shape.Dims(i);
    }
    return prod;
  };

  const int batch_dim0 =
      broadcast_dim(extended_lhs_shape.Dims(0), extended_rhs_shape.Dims(0));
  const int batch_dim1 =
      broadcast_dim(extended_lhs_shape.Dims(1), extended_rhs_shape.Dims(1));
  const int batch_dim2 =
      broadcast_dim(extended_lhs_shape.Dims(2), extended_rhs_shape.Dims(2));

  const int lhs_ext0 = extent(extended_lhs_shape, 0);
  const int lhs_ext1 = extent(extended_lhs_shape, 1);
  const int lhs_ext2 = extent(extended_lhs_shape, 2);
  const int rhs_ext0 = extent(extended_rhs_shape, 0);
  const int rhs_ext1 = extent(extended_rhs_shape, 1);
  const int rhs_ext2 = extent(extended_rhs_shape, 2);

  // Set params for each matrix multiply.
  const int lhs_rows = extended_lhs_shape.Dims(3);
  const int rhs_cols = extended_rhs_shape.Dims(4);
  const int accum_depth = extended_lhs_shape.Dims(4);

  const int ioff_ext0 = rhs_ext0 == 0 ? 0 : rhs_cols;
  const int ioff_ext1 = rhs_ext1 == 0 ? 0 : rhs_cols;
  const int ioff_ext2 = rhs_ext2 == 0 ? 0 : rhs_cols;
  const int woff_ext0 = lhs_ext0 == 0 ? 0 : lhs_rows;
  const int woff_ext1 = lhs_ext1 == 0 ? 0 : lhs_rows;
  const int woff_ext2 = lhs_ext2 == 0 ? 0 : lhs_rows;

  if (!compute_row_sums || *compute_row_sums) {
    int num_weights_matrices = 1;
    for (int i = 1; i < extended_lhs_shape.DimensionsCount() - 2; ++i) {
      num_weights_matrices *= extended_lhs_shape.Dims(i);
    }
    tensor_utils::ReductionSumVector(
        lhs_data, row_sums, num_weights_matrices * lhs_rows, accum_depth);
    if (compute_row_sums) {
      *compute_row_sums = false;
    }
  }

  for (int b0 = 0; b0 < batch_dim0; ++b0) {
    const int8_t* lhs_ptr0 = lhs_data + (b0 * lhs_ext0);
    const int8_t* rhs_ptr0 = rhs_data + (b0 * rhs_ext0);
    const int32_t* ioff_ptr0 = input_offset + (b0 * ioff_ext0);
    const float* scale_ptr0 = scaling_factors + (b0 * ioff_ext0);
    int32_t* woff_ptr0 = row_sums + (b0 * woff_ext0);
    for (int b1 = 0; b1 < batch_dim1; ++b1) {
      const int8_t* lhs_ptr1 = lhs_ptr0 + b1 * lhs_ext1;
      const int8_t* rhs_ptr1 = rhs_ptr0 + b1 * rhs_ext1;
      const int32_t* ioff_ptr1 = ioff_ptr0 + (b1 * ioff_ext1);
      const float* scale_ptr1 = scale_ptr0 + (b1 * ioff_ext1);
      int32_t* woff_ptr1 = woff_ptr0 + (b1 * woff_ext1);
      for (int b2 = 0; b2 < batch_dim2; ++b2) {
        const int8_t* lhs_ptr2 = lhs_ptr1 + b2 * lhs_ext2;
        const int8_t* rhs_ptr2 = rhs_ptr1 + b2 * rhs_ext2;
        const int32_t* ioff_ptr2 = ioff_ptr1 + (b2 * ioff_ext2);
        const float* scale_ptr2 = scale_ptr1 + (b2 * ioff_ext2);
        int32_t* woff_ptr2 = woff_ptr1 + (b2 * woff_ext2);
        float* out_ptr = output_data + ((b0 * batch_dim1 * batch_dim2) +
                                        b1 * batch_dim2 + b2) *
                                           lhs_rows * rhs_cols;
        tensor_utils::MatrixBatchVectorMultiplyAccumulate(
            lhs_ptr2, lhs_rows, accum_depth, rhs_ptr2, scale_ptr2, rhs_cols,
            out_ptr, /*per_channel_scale=*/nullptr, ioff_ptr2, accum_scratch,
            woff_ptr2, compute_row_sums, context);
      }
    }
  }
}

inline void BatchMatMul(const FullyConnectedParams& params,
                        const RuntimeShape& lhs_shape, const int8_t* lhs_data,
                        const RuntimeShape& rhs_shape, const int8_t* rhs_data,
                        const RuntimeShape& output_shape, int8_t* output_data,
                        CpuBackendContext* context,
                        bool transpose_lhs = false) {
  using ::tflite::cpu_backend_gemm::Gemm;
  using ::tflite::cpu_backend_gemm::GemmParams;
  using ::tflite::cpu_backend_gemm::MatrixParams;

  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(5, rhs_shape);

  // Determine which dimension is the broadcast dimension.
  auto broadcast_dim = [](int lhs_dim, int rhs_dim) {
    if (lhs_dim == rhs_dim) return lhs_dim;
    if (lhs_dim == 1) return rhs_dim;
    TFLITE_DCHECK_EQ(rhs_dim, 1);
    return lhs_dim;
  };

  // Compute the "extent" for iterating on this dimension.
  // If we are broadcasting, then don't advance (i.e return 0).
  auto extent = [](const RuntimeShape& shape, int x) {
    if (shape.Dims(x) == 1) {
      return 0;
    }
    int prod = 1;
    for (int i = x + 1; i < shape.DimensionsCount(); ++i) {
      prod *= shape.Dims(i);
    }
    return prod;
  };

  const int batch_dim0 =
      broadcast_dim(extended_lhs_shape.Dims(0), extended_rhs_shape.Dims(0));
  const int batch_dim1 =
      broadcast_dim(extended_lhs_shape.Dims(1), extended_rhs_shape.Dims(1));
  const int batch_dim2 =
      broadcast_dim(extended_lhs_shape.Dims(2), extended_rhs_shape.Dims(2));

  const int lhs_ext0 = extent(extended_lhs_shape, 0);
  const int lhs_ext1 = extent(extended_lhs_shape, 1);
  const int lhs_ext2 = extent(extended_lhs_shape, 2);
  const int rhs_ext0 = extent(extended_rhs_shape, 0);
  const int rhs_ext1 = extent(extended_rhs_shape, 1);
  const int rhs_ext2 = extent(extended_rhs_shape, 2);

  // Set params for each matrix multiply.
  const int lhs_rows = extended_lhs_shape.Dims(3);
  const int rhs_cols = extended_rhs_shape.Dims(4);
  const int accum_depth = extended_lhs_shape.Dims(4);

  const int32 input_offset = params.input_offset;
  const int32 filter_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  MatrixParams<int8_t> lhs_params;
  if (transpose_lhs) {
    lhs_params.order = cpu_backend_gemm::Order::kColMajor;
  } else {
    lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  }
  lhs_params.rows = lhs_rows;
  lhs_params.cols = accum_depth;
  lhs_params.zero_point = -filter_offset;

  MatrixParams<int8_t> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = accum_depth;
  rhs_params.cols = rhs_cols;
  rhs_params.zero_point = -input_offset;

  MatrixParams<int8_t> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = lhs_rows;
  dst_params.cols = rhs_cols;
  dst_params.zero_point = output_offset;

  for (int b0 = 0; b0 < batch_dim0; ++b0) {
    const int8_t* lhs_ptr0 = lhs_data + (b0 * lhs_ext0);
    const int8_t* rhs_ptr0 = rhs_data + (b0 * rhs_ext0);
    for (int b1 = 0; b1 < batch_dim1; ++b1) {
      const int8_t* lhs_ptr1 = lhs_ptr0 + b1 * lhs_ext1;
      const int8_t* rhs_ptr1 = rhs_ptr0 + b1 * rhs_ext1;
      for (int b2 = 0; b2 < batch_dim2; ++b2) {
        const int8_t* lhs_ptr2 = lhs_ptr1 + b2 * lhs_ext2;
        const int8_t* rhs_ptr2 = rhs_ptr1 + b2 * rhs_ext2;
        int8_t* out_ptr = output_data + ((b0 * batch_dim1 * batch_dim2) +
                                         b1 * batch_dim2 + b2) *
                                            lhs_rows * rhs_cols;

        GemmParams<int32_t, int8_t> gemm_params;
        gemm_params.clamp_min = output_activation_min;
        gemm_params.clamp_max = output_activation_max;
        gemm_params.multiplier_fixedpoint = output_multiplier;
        gemm_params.multiplier_exponent = output_shift;
        cpu_backend_gemm::Gemm(lhs_params, lhs_ptr2, rhs_params, rhs_ptr2,
                               dst_params, out_ptr, gemm_params, context);
      }
    }
  }
}

inline void BatchMatMul(const FullyConnectedParams& params,
                        const RuntimeShape& lhs_shape, const int8_t* lhs_data,
                        const RuntimeShape& rhs_shape, const int8_t* rhs_data,
                        const RuntimeShape& output_shape, int32_t* output_data,
                        CpuBackendContext* context,
                        bool transpose_lhs = false) {
  using ::tflite::cpu_backend_gemm::Gemm;
  using ::tflite::cpu_backend_gemm::GemmParams;
  using ::tflite::cpu_backend_gemm::MatrixParams;

  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(5, lhs_shape);
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(5, rhs_shape);

  // Determine which dimension is the broadcast dimension.
  auto broadcast_dim = [](int lhs_dim, int rhs_dim) {
    if (lhs_dim == rhs_dim) return lhs_dim;
    if (lhs_dim == 1) return rhs_dim;
    TFLITE_DCHECK_EQ(rhs_dim, 1);
    return lhs_dim;
  };

  // Compute the "extent" for iterating on this dimension.
  // If we are broadcasting, then don't advance (i.e return 0).
  auto extent = [](const RuntimeShape& shape, int x) {
    if (shape.Dims(x) == 1) {
      return 0;
    }
    int prod = 1;
    for (int i = x + 1; i < shape.DimensionsCount(); ++i) {
      prod *= shape.Dims(i);
    }
    return prod;
  };

  const int batch_dim0 =
      broadcast_dim(extended_lhs_shape.Dims(0), extended_rhs_shape.Dims(0));
  const int batch_dim1 =
      broadcast_dim(extended_lhs_shape.Dims(1), extended_rhs_shape.Dims(1));
  const int batch_dim2 =
      broadcast_dim(extended_lhs_shape.Dims(2), extended_rhs_shape.Dims(2));

  const int lhs_ext0 = extent(extended_lhs_shape, 0);
  const int lhs_ext1 = extent(extended_lhs_shape, 1);
  const int lhs_ext2 = extent(extended_lhs_shape, 2);
  const int rhs_ext0 = extent(extended_rhs_shape, 0);
  const int rhs_ext1 = extent(extended_rhs_shape, 1);
  const int rhs_ext2 = extent(extended_rhs_shape, 2);

  // Set params for each matrix multiply.
  const int lhs_rows = extended_lhs_shape.Dims(3);
  const int rhs_cols = extended_rhs_shape.Dims(4);
  const int accum_depth = extended_lhs_shape.Dims(4);

  const int32 input_offset = params.input_offset;
  const int32 weights_offset = params.weights_offset;
  const int32 output_offset = params.output_offset;
  const int32 output_multiplier = params.output_multiplier;
  const int output_shift = params.output_shift;
  const int32 output_activation_min = params.quantized_activation_min;
  const int32 output_activation_max = params.quantized_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);

  MatrixParams<int8_t> lhs_params;
  if (transpose_lhs) {
    lhs_params.order = cpu_backend_gemm::Order::kColMajor;
  } else {
    lhs_params.order = cpu_backend_gemm::Order::kRowMajor;
  }
  lhs_params.rows = lhs_rows;
  lhs_params.cols = accum_depth;
  lhs_params.zero_point = -weights_offset;

  MatrixParams<int8_t> rhs_params;
  rhs_params.order = cpu_backend_gemm::Order::kColMajor;
  rhs_params.rows = accum_depth;
  rhs_params.cols = rhs_cols;
  rhs_params.zero_point = -input_offset;

  MatrixParams<int32_t> dst_params;
  dst_params.order = cpu_backend_gemm::Order::kColMajor;
  dst_params.rows = lhs_rows;
  dst_params.cols = rhs_cols;
  dst_params.zero_point = output_offset;

  for (int b0 = 0; b0 < batch_dim0; ++b0) {
    const int8_t* lhs_ptr0 = lhs_data + (b0 * lhs_ext0);
    const int8_t* rhs_ptr0 = rhs_data + (b0 * rhs_ext0);
    for (int b1 = 0; b1 < batch_dim1; ++b1) {
      const int8_t* lhs_ptr1 = lhs_ptr0 + b1 * lhs_ext1;
      const int8_t* rhs_ptr1 = rhs_ptr0 + b1 * rhs_ext1;
      for (int b2 = 0; b2 < batch_dim2; ++b2) {
        const int8_t* lhs_ptr2 = lhs_ptr1 + b2 * lhs_ext2;
        const int8_t* rhs_ptr2 = rhs_ptr1 + b2 * rhs_ext2;
        int32_t* out_ptr = output_data + ((b0 * batch_dim1 * batch_dim2) +
                                          b1 * batch_dim2 + b2) *
                                             lhs_rows * rhs_cols;

        GemmParams<int32_t, int32_t> gemm_params;
        gemm_params.clamp_min = output_activation_min;
        gemm_params.clamp_max = output_activation_max;
        gemm_params.multiplier_fixedpoint = output_multiplier;
        gemm_params.multiplier_exponent = output_shift;
        cpu_backend_gemm::Gemm(lhs_params, lhs_ptr2, rhs_params, rhs_ptr2,
                               dst_params, out_ptr, gemm_params, context);
      }
    }
  }
}

}  // namespace optimized_ops
}  // namespace tflite

#endif  // TENSORFLOW_LITE_KERNELS_INTERNAL_OPTIMIZED_BATCH_MATMUL_H_
