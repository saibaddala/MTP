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

#ifndef TENSORFLOW_CORE_KERNELS_BATCH_KERNEL_TEST_UTIL_H_
#define TENSORFLOW_CORE_KERNELS_BATCH_KERNEL_TEST_UTIL_H_

#include "node_def_builder.h"
#include "batch_kernels.h"
#include "ops_testutil.h"
#include "ops_util.h"
#include "status_test_util.h"
#include "env.h"
#include "test.h"

namespace tensorflow {
namespace internal {
class BatchFunctionKernelTestAccess {
 public:
  explicit BatchFunctionKernelTestAccess(BatchFunctionKernel* kernel);

  bool enable_adaptive_batch_threads() const;

 private:
  BatchFunctionKernel* const kernel_;
};

}  // namespace internal

class BatchFunctionKernelTestBase : public OpsTestBase,
                                    public ::testing::WithParamInterface<bool> {
 public:
  bool enable_adaptive_scheduler() const;

  // Init test fixture with a batch kernel instance.
  Status Init();
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_KERNELS_BATCH_KERNEL_TEST_UTIL_H_
