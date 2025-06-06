// Copyright 2022 The Abseil Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ABSL_STRINGS_INTERNAL_STRINGIFY_SINK_H_
#define ABSL_STRINGS_INTERNAL_STRINGIFY_SINK_H_

#include <string>
#include <type_traits>
#include <utility>

#include "string_view.h"

namespace absl {
ABSL_NAMESPACE_BEGIN

namespace strings_internal {
class StringifySink {
 public:
  void Append(size_t count, char ch);

  void Append(string_view v);

  // Support `absl::Format(&sink, format, args...)`.
  friend void AbslFormatFlush(StringifySink* sink, absl::string_view v) {
    sink->Append(v);
  }

 private:
  template <typename T>
  friend string_view ExtractStringification(StringifySink& sink, const T& v);

  std::string buffer_;
};

template <typename T>
string_view ExtractStringification(StringifySink& sink, const T& v) {
  AbslStringify(sink, v);
  return sink.buffer_;
}

}  // namespace strings_internal

ABSL_NAMESPACE_END
}  // namespace absl

#endif  // ABSL_STRINGS_INTERNAL_STRINGIFY_SINK_H_
