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

#ifndef TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_OPENXLA_CONVERSION_CONVERT_LIBRARY_OPS_H_
#define TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_OPENXLA_CONVERSION_CONVERT_LIBRARY_OPS_H_

#include "PatternMatch.h"  // from @llvm-project
#include "DialectConversion.h"  // from @llvm-project
#include "de_bufferization.h"
#include "xla_gpu_api.h"

namespace xla {
namespace gpu {

// Appends patterns to convert `lmhlo_gpu` operations corresponding to library
// calls (cuBLAS, cuDNN, etc. operations).
void populateLibraryOpsConversionPatterns(mlir::RewritePatternSet &patterns,
                                          mlir::TypeConverter &converter,
                                          DeBufferization &state,
                                          XlaGpuApi &api);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_MLIR_BACKENDS_OPENXLA_CONVERSION_CONVERT_LIBRARY_OPS_H_
