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
#ifndef TENSORFLOW_CORE_PROFILER_UTILS_TFSTREAMZ_UTILS_H_
#define TENSORFLOW_CORE_PROFILER_UTILS_TFSTREAMZ_UTILS_H_

#include <memory>
#include <vector>

#include "collected_metrics.h"
#include "status.h"
#include "types.h"
#include "xplane.pb.h"

namespace tensorflow {
namespace profiler {

struct TfStreamzSnapshot {
  std::unique_ptr<monitoring::CollectedMetrics> metrics;
  uint64 start_time_ns;  // time before collection.
  uint64 end_time_ns;    // time after collection.
};

Status SerializeToXPlane(const std::vector<TfStreamzSnapshot>& snapshots,
                         XPlane* plane, uint64 line_start_time_ns);

}  // namespace profiler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PROFILER_UTILS_TFSTREAMZ_UTILS_H_
