// Copyright 2021 The Abseil Authors
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

#ifndef ABSL_STRINGS_CORD_ANALYSIS_H_
#define ABSL_STRINGS_CORD_ANALYSIS_H_

#include <cstddef>
#include <cstdint>

#include "config.h"
#include "cord_internal.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace cord_internal {

// Returns the *approximate* number of bytes held in full or in part by this
// Cord (which may not remain the same between invocations). Cords that share
// memory could each be "charged" independently for the same shared memory.
size_t GetEstimatedMemoryUsage(const CordRep* rep);

// Returns the *approximate* number of bytes held in full or in part by this
// CordRep weighted by the sharing ratio of that data. For example, if some data
// edge is shared by 4 different Cords, then each cord is attribute 1/4th of
// the total memory usage as a 'fair share' of the total memory usage.
size_t GetEstimatedFairShareMemoryUsage(const CordRep* rep);

}  // namespace cord_internal
ABSL_NAMESPACE_END
}  // namespace absl


#endif  // ABSL_STRINGS_CORD_ANALYSIS_H_
