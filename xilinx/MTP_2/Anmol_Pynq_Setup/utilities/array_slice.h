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

#ifndef TENSORFLOW_CORE_LIB_GTL_ARRAY_SLICE_H_
#define TENSORFLOW_CORE_LIB_GTL_ARRAY_SLICE_H_

#include "span.h"
// TODO(timshen): This is kept only because lots of targets transitively depend
// on it. Remove all targets' dependencies.
#include "inlined_vector.h"

namespace tensorflow {
namespace gtl {

template <typename T>
using ArraySlice = absl::Span<const T>;

template <typename T>
using MutableArraySlice = absl::Span<T>;

}  // namespace gtl
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_LIB_GTL_ARRAY_SLICE_H_
