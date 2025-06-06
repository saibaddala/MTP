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

//===- kernel_creator.h -----------------------------------------*- C++ -*-===//
//
// This file declares the function to compile a TF kernel function to gpu
// binary (hsaco for AMD, cubin for NVIDIA) or to a gpu binary with host side.
//
//===----------------------------------------------------------------------===//
#ifndef TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_KERNEL_CREATOR_H_
#define TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_KERNEL_CREATOR_H_

#include <utility>

#include "ArrayRef.h"
#include "StringRef.h"
#include "FuncOps.h"  // from @llvm-project
#include "BuiltinOps.h"  // from @llvm-project
#include "MLIRContext.h"  // from @llvm-project
#include "statusor.h"

namespace tensorflow {
namespace kernel_gen {

// Parses tf_code to create a module. An MLIRContext is taken in case any
// unexpected dialects are needed.
StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> SetupContextAndParseModule(
    mlir::MLIRContext& context, llvm::StringRef tf_code);

// Converts TF code to LLVM with or without GPU support.
StatusOr<mlir::OwningOpRef<mlir::ModuleOp>> GenerateKernelForTfCode(
    mlir::MLIRContext& context, llvm::StringRef tf_code,
    llvm::ArrayRef<std::string> architectures,
    llvm::ArrayRef<int64_t> tile_sizes, llvm::ArrayRef<int64_t> unroll_factors,
    int64_t max_supported_rank, bool print_ptx, bool print_llvmir,
    bool enable_ftz, bool index_64bit, bool jit_compile,
    bool jit_i64_indexed_for_large_tensors, bool apply_cl_options);

}  // namespace kernel_gen
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_MLIR_TOOLS_KERNEL_GEN_KERNEL_CREATOR_H_
