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

#ifndef TENSORFLOW_LITE_DELEGATES_XNNPACK_UNARY_ELEMENTWISE_TESTER_H_
#define TENSORFLOW_LITE_DELEGATES_XNNPACK_UNARY_ELEMENTWISE_TESTER_H_

#include <cstdint>
#include <vector>

#include <gtest.h>
#include "common.h"
#include "schema_generated.h"

namespace tflite {
namespace xnnpack {

class UnaryElementwiseTester {
 public:
  UnaryElementwiseTester() = default;
  UnaryElementwiseTester(const UnaryElementwiseTester&) = delete;
  UnaryElementwiseTester& operator=(const UnaryElementwiseTester&) = delete;

  inline UnaryElementwiseTester& Shape(std::initializer_list<int32_t> shape) {
    for (auto it = shape.begin(); it != shape.end(); ++it) {
      EXPECT_GT(*it, 0);
    }
    shape_ = std::vector<int32_t>(shape.begin(), shape.end());
    size_ = UnaryElementwiseTester::ComputeSize(shape_);
    return *this;
  }

  const std::vector<int32_t>& Shape() const { return shape_; }

  int32_t Size() const { return size_; }

  inline UnaryElementwiseTester& RelativeTolerance(float relative_tolerance) {
    relative_tolerance_ = relative_tolerance;
    return *this;
  }

  float RelativeTolerance() const { return relative_tolerance_; }

  void Test(tflite::BuiltinOperator unary_op, TfLiteDelegate* delegate) const;

 private:
  std::vector<char> CreateTfLiteModel(tflite::BuiltinOperator unary_op) const;

  static int32_t ComputeSize(const std::vector<int32_t>& shape);

  std::vector<int32_t> shape_;
  int32_t size_;
  float relative_tolerance_{10.0f};
};

}  // namespace xnnpack
}  // namespace tflite

#endif  // TENSORFLOW_LITE_DELEGATES_XNNPACK_UNARY_ELEMENTWISE_TESTER_H_
