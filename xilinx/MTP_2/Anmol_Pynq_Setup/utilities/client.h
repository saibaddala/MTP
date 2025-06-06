/* Copyright 2020 Google LLC

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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_CLIENT_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_CLIENT_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "time.h"
#include "channel.h"
#include "protocol.grpc.pb.h"
#include "statusor.h"
#include "types.h"
#include "env.h"

namespace tsl {
class CoordinationServiceAgent;
}  // namespace tsl

namespace xla {

class DistributedRuntimeClient {
 public:
  struct Options {
    // This node's global ID. Required.
    int32_t node_id = -1;

    // Environment used for starting threads.
    tsl::Env* env = tsl::Env::Default();

    // RPC timeout used for RPC that don't have their own timeouts.
    absl::Duration rpc_timeout = absl::Seconds(120);

    // Time period for which Connect() should be retried. The client will keep
    // trying to open the initial connection for this period, even if any
    // individual Connect() RPC fails. May be zero, in which case Connect() will
    // only be attempted once.
    absl::Duration init_timeout = absl::ZeroDuration();

    // How long to wait for all nodes to call Shutdown(). If the timeout
    // expires, then shutdown() reports an error and returns control.
    absl::Duration shutdown_timeout = absl::Minutes(5);

    // Interval at which the client should send heartbeat RPCs to the
    // coordinator.
    absl::Duration heartbeat_interval = absl::Seconds(10);

    // How many failed heartbeat RPCs may fail due to a possibly-ephemeral
    // reason before we decide the coordinator has vanished and that we should
    // shut down.
    int max_missing_heartbeats = 10;

    // Callback invoked by the client when notification of a missing heartbeat
    // is reported by the coordinator, or we have not heard from the coordinator
    // recently. `coordinator_reported_failure` is true in the former case.
    // Exposed so tests can override this behavior to something non-fatal.
    std::function<void(xla::Status, bool coordinator_reported_failure)>
        missed_heartbeat_callback =
            [](xla::Status status, bool coordinator_reported_failure) {
              if (coordinator_reported_failure) {
                LOG(QFATAL)
                    << "Terminating process because the coordinator detected "
                       "missing heartbeats. This most likely indicates that "
                       "another task died; see the other task logs for more "
                       "details. Disable Python buffering, i.e. `python -u`, "
                       "to be sure to see all the previous output. Status: "
                    << status;
              } else {
                LOG(QFATAL)
                    << "Terminating process because of missing heartbeat "
                       "response from the coordinator. This most likely "
                       "indicates that the coordinator task died; see the "
                       "coordinator's task logs for more details. "
                       "Disable Python buffering, i.e. `python -u`, to be "
                       "sure to see all the previous output. Status: "
                    << status;
              }
            };

    // For testing. Should the client explicitly Shutdown() on destruction?
    bool shutdown_on_destruction = true;
  };

  virtual ~DistributedRuntimeClient() = default;

  // Connects to the master, and blocks until all clients have successfully
  // connected.
  // Not thread-safe, i.e., calls to Connect()/Shutdown()/EnumerateDevices()
  // must be serialized by some other means.
  virtual xla::Status Connect() = 0;

  // Reports to the master that the client is ready to shutdown, and blocks
  // until all clients are ready to shutdown or the shutdown timeout expires.
  // Not thread-safe.
  virtual xla::Status Shutdown() = 0;

  // Blocking enumeration of global devices. Used by the GPU platform.
  // Not thread-safe.
  virtual xla::Status EnumerateDevices(
      const LocalTopologyProto& local_topology,
      GlobalTopologyProto* global_topology) = 0;

  // The following APIs are thread-safe.

  // Key-value store API.
  // There are no concurrency guarantees. To avoid a race / impose an ordering
  // on potentially concurrent ops (e.g. set, delete), use WaitAtBarrier().
  virtual xla::StatusOr<std::string> BlockingKeyValueGet(
      std::string key, absl::Duration timeout) = 0;

  // Get all key-value pairs under a directory (key).
  // A value is considered to be in the directory if its key is prefixed with
  // the directory.
  // This is not a blocking call. If no keys are found, an empty vector is
  // returned immediately.
  virtual xla::StatusOr<std::vector<std::pair<std::string, std::string>>>
  KeyValueDirGet(absl::string_view key) = 0;

  virtual xla::Status KeyValueSet(std::string key, std::string value) = 0;

  // Delete the key-value. If the key is a directory, recursively clean
  // up all key-values under the directory.
  virtual xla::Status KeyValueDelete(std::string key) = 0;

  // Blocks until all nodes are at the barrier or the barrier times out.
  // `barrier_id` should be unique across barriers.
  virtual xla::Status WaitAtBarrier(std::string barrier_id,
                                    absl::Duration timeout) = 0;

  // Returns pointer to coordination service agent, or InternalError if the
  // client does not use coordination service.
  virtual StatusOr<tsl::CoordinationServiceAgent*>
  GetCoordinationServiceAgent() = 0;
};

// Creates a distributed runtime client.
std::unique_ptr<DistributedRuntimeClient> GetDistributedRuntimeClient(
    std::shared_ptr<::grpc::Channel> channel,
    const DistributedRuntimeClient::Options& options,
    bool use_coordination_service);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_CLIENT_H_
