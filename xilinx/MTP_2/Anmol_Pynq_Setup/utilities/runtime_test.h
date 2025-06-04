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
#ifndef TENSORFLOW_CC_EXPERIMENTAL_LIBTF_TESTS_RUNTIME_TEST_H_
#define TENSORFLOW_CC_EXPERIMENTAL_LIBTF_TESTS_RUNTIME_TEST_H_

#include "status.h"
#include "match.h"
#include "unified_api_testutil.h"
#include "tf_datatype.h"
#include "tf_status_helper.h"
#include "object.h"
#include "runtime.h"
#include "value.h"
#include "status_test_util.h"
#include "resource_loader.h"
#include "status_matchers.h"
#include "statusor.h"
#include "test.h"

namespace tf {
namespace libtf {
namespace runtime {

typedef Runtime (*RuntimeFn)();

class RuntimeTest : public ::testing::TestWithParam<RuntimeFn> {};

}  // namespace runtime
}  // namespace libtf
}  // namespace tf

#endif  // TENSORFLOW_CC_EXPERIMENTAL_LIBTF_TESTS_RUNTIME_TEST_H_
