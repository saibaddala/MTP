/* Copyright 2017 The TensorFlow Authors All Rights Reserved.

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
// GRPC client to perform on-demand profiling

#ifndef TENSORFLOW_TSL_PROFILER_RPC_CLIENT_CAPTURE_PROFILE_H_
#define TENSORFLOW_TSL_PROFILER_RPC_CLIENT_CAPTURE_PROFILE_H_

#include <string>
#include <variant>

#include "flat_hash_map.h"
#include "status.h"
#include "profiler_options.pb.h"
#include "profiler_service.pb.h"
#include "xplane.pb.h"

namespace tsl {
namespace profiler {

// Convert XSpace to tool data and saves under <logdir>/plugins/profile/.
Status ExportToTensorBoard(const tensorflow::profiler::XSpace& xspace,
                           const std::string& logdir,
                           bool also_export_trace_json = false);

// Collects one sample of monitoring profile and shows user-friendly metrics.
// If timestamp flag is true, timestamp will be displayed in "%H:%M:%S" format.
Status Monitor(const std::string& service_addr, int duration_ms,
               int monitoring_level, bool display_timestamp,
               std::string* result);

// Starts tracing on a single or multiple hosts. Each host will save the result
// in the given logdir. If no trace was collected, retries tracing for
// num_tracing_attempts. Assumes that options have been validated.
Status CaptureRemoteTrace(const std::string& logdir, int num_tracing_attempts,
                          tensorflow::RemoteProfilerSessionManagerOptions& opts,
                          bool is_cloud_tpu_session);

// Generates RemoteProfilerSessionManagerOptions from inputs and calls
// CaptureRemoteTrace.
Status CaptureRemoteTrace(
    const char* service_addr, const char* logdir, const char* worker_list,
    bool include_dataset_ops, int duration_ms, int num_tracing_attempts,
    const absl::flat_hash_map<std::string, std::variant<int, std::string>>&
        options);

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_RPC_CLIENT_CAPTURE_PROFILE_H_
