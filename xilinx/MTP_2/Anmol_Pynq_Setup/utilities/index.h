/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_INDEX_H_
#define TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_INDEX_H_

#include <cstdint>
#include <ostream>
#include <string>

#include "inlined_vector.h"
#include "span.h"
#include "logging.h"

namespace xla {
namespace ifrt {

// Multi-dimensional index. Every element must be equal to or greater than 0.
class Index {
 public:
  // Maximum elements to inline.
  static constexpr int kInlineElementSize = 6;

  using Elements = absl::InlinedVector<int64_t, kInlineElementSize>;

  explicit Index(absl::Span<const int64_t> elements)
      : elements_(Elements(elements.begin(), elements.end())) {}

  static Index Zeros(int num_elements) {
    return Index(Elements(/*n=*/num_elements));
  }

  Index(const Index&) = default;
  Index(Index&&) = default;
  Index& operator=(const Index&) = default;
  Index& operator=(Index&&) = default;

  absl::Span<const int64_t> elements() const { return elements_; }

  bool operator==(const Index& other) const {
    return elements_ == other.elements_;
  }
  bool operator!=(const Index& other) const {
    return elements_ != other.elements_;
  }
  Index operator+(const Index& offset) const {
    CHECK_EQ(elements_.size(), offset.elements_.size());
    Index result = *this;
    for (int i = 0; i < elements_.size(); ++i) {
      result.elements_[i] += offset.elements_[i];
    }
    return result;
  }
  Index operator-(const Index& offset) const {
    CHECK_EQ(elements_.size(), offset.elements_.size());
    Index result = *this;
    for (int i = 0; i < elements_.size(); ++i) {
      result.elements_[i] -= offset.elements_[i];
    }
    return result;
  }
  Index operator*(absl::Span<const int64_t> multiplier) const {
    CHECK_EQ(elements_.size(), multiplier.size());
    Index result = *this;
    for (int i = 0; i < elements_.size(); ++i) {
      result.elements_[i] *= multiplier[i];
    }
    return result;
  }
  Index& operator+=(const Index& offset) { return *this = *this + offset; }
  Index& operator-=(const Index& offset) { return *this = *this - offset; }
  Index& operator*=(absl::Span<const int64_t> multiplier) {
    return *this = *this * multiplier;
  }

  std::string DebugString() const;

 private:
  Elements elements_;
};

std::ostream& operator<<(std::ostream& os, const Index& index);

}  // namespace ifrt
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PYTHON_IFRT_INDEX_H_
