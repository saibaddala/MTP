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

#ifndef TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_SERVICE_H_
#define TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_SERVICE_H_

#include <memory>
#include <string>

#include "flat_hash_map.h"
#include "mutex.h"
#include "notification.h"
#include "time.h"
#include "server_credentials.h"
#include "key_value_store.h"
#include "protocol.grpc.pb.h"
#include "statusor.h"
#include "types.h"
#include "coordination_service.h"
#include "async_service_interface.h"
#include "env.h"
#include "threadpool.h"

namespace xla {

typedef int NodeId;

class DistributedRuntimeServiceImpl final
    : public grpc::DistributedRuntimeService::Service {
 public:
  struct Options {
    // Number of nodes in the job. Mandatory. Must be non-negative.
    int num_nodes = -1;

    tsl::Env* env = tsl::Env::Default();

    // Interval at which the service should check for missed heartbeat RPCs
    // from the clients.
    absl::Duration heartbeat_interval = absl::Seconds(10);

    // Number of heartbeats that a client may miss in a row before the
    // coordinator concludes that a client has vanished.
    int max_missing_heartbeats = 10;

    // How long should we wait for all clients to call EnumerateDevices() before
    // giving up?
    absl::Duration enumerate_devices_timeout = absl::Seconds(60);

    // How long should we wait for all clients to call Shutdown() before giving
    // up and returning a failure?
    absl::Duration shutdown_timeout = absl::Minutes(5);
  };
  explicit DistributedRuntimeServiceImpl(const Options& options);
  ~DistributedRuntimeServiceImpl() override;

  DistributedRuntimeServiceImpl(const DistributedRuntimeServiceImpl&) = delete;
  DistributedRuntimeServiceImpl(DistributedRuntimeServiceImpl&&) = delete;
  DistributedRuntimeServiceImpl& operator=(
      const DistributedRuntimeServiceImpl&) = delete;
  DistributedRuntimeServiceImpl&& operator=(DistributedRuntimeServiceImpl&&) =
      delete;

  ::grpc::Status Connect(::grpc::ServerContext* context,
                         const ConnectRequest* request,
                         ConnectResponse* response) override;

  ::grpc::Status Shutdown(::grpc::ServerContext* context,
                          const ShutdownRequest* request,
                          ShutdownResponse* response) override;

  ::grpc::Status Heartbeat(::grpc::ServerContext* context,
                           const HeartbeatRequest* request,
                           HeartbeatResponse* response) override;

  ::grpc::Status EnumerateDevices(::grpc::ServerContext* context,
                                  const EnumerateDevicesRequest* request,
                                  EnumerateDevicesResponse* response) override;

  ::grpc::Status KeyValueGet(::grpc::ServerContext* context,
                             const KeyValueGetRequest* request,
                             KeyValueGetResponse* response) override;

  ::grpc::Status KeyValueSet(::grpc::ServerContext* context,
                             const KeyValueSetRequest* request,
                             KeyValueSetResponse* response) override;

  ::grpc::Status WaitAtBarrier(::grpc::ServerContext* context,
                               const WaitAtBarrierRequest* request,
                               WaitAtBarrierResponse* response) override;

 private:
  // Entry point for the heartbeat checking thread.
  void HeartbeatLoop();

  // Validates a session id number matches the current session id.
  xla::Status ValidateSessionId(uint64_t session_id);

  // Validates a node id number.
  xla::Status ValidateNodeId(int node_id);

  const Options options_;
  const uint64_t session_id_;

  absl::Mutex mu_;
  enum class State { kInitializing, kRunning, kClosed };
  State state_ ABSL_GUARDED_BY(mu_) = State::kInitializing;
  Status service_status_ ABSL_GUARDED_BY(mu_);

  // State for Connect() and heartbeats.
  struct Node {
    // Have we heard from a task with this ID?
    bool present = false;

    // A unique ID belonging to the client. Used to identify the client that
    // most recently called Connect() with a particular task id.
    uint64_t client_id = 0;

    // When did we last receive a heartbeat from this task?
    absl::Time last_heartbeat = absl::InfinitePast();
  };
  int num_nodes_present_ ABSL_GUARDED_BY(mu_) = 0;
  std::vector<Node> nodes_ ABSL_GUARDED_BY(mu_);

  // State for EnumerateDevices.
  int num_topologies_present_ ABSL_GUARDED_BY(mu_) = 0;
  std::vector<LocalTopologyProto> local_topologies_ ABSL_GUARDED_BY(mu_);
  std::optional<GlobalTopologyProto> topology_ ABSL_GUARDED_BY(mu_);

  // State for Shutdown(). Counter of how many nodes are blocked at the
  // Shutdown() barrier.
  int num_nodes_shutting_down_ ABSL_GUARDED_BY(mu_) = 0;

  // This dictionary tracks the number of nodes per barrier.
  absl::flat_hash_map<std::string, int> barrier_id_to_num_nodes_
      ABSL_GUARDED_BY(mu_);

  // Key-value store, used by distributed GPU code to share NCCL state.
  KeyValueStore key_value_store_;

  // Notification that tells the heartbeat thread to stop.
  absl::Notification stop_heartbeat_thread_;

  // Thread that checks for missing hearbeats from the clients periodically.
  std::unique_ptr<tsl::Thread> heartbeat_thread_;
};

class CoordinationServiceImpl {
 public:
  CoordinationServiceImpl(const DistributedRuntimeServiceImpl::Options& options,
                          ::grpc::ServerBuilder* builder);
  ~CoordinationServiceImpl();

  // Must be called after gRPC server has started.
  void StartRpcThread();

  CoordinationServiceImpl(const CoordinationServiceImpl&) = delete;
  CoordinationServiceImpl(CoordinationServiceImpl&&) = delete;
  CoordinationServiceImpl& operator=(const CoordinationServiceImpl&) = delete;
  CoordinationServiceImpl&& operator=(CoordinationServiceImpl&&) = delete;

 private:
  tsl::Env* env_ = nullptr;  // Not owned.
  std::unique_ptr<tsl::CoordinationServiceInterface> coord_service_;
  std::unique_ptr<tsl::thread::ThreadPool> coord_compute_pool_;
  std::unique_ptr<tsl::AsyncServiceInterface> coord_rpc_service_;
  std::unique_ptr<tsl::Thread> coord_rpc_thread_;
};

class DistributedRuntimeService {
 public:
  static xla::StatusOr<std::unique_ptr<DistributedRuntimeService>> Get(
      const std::string& address,
      std::shared_ptr<::grpc::ServerCredentials> credentials,
      const DistributedRuntimeServiceImpl::Options& options,
      bool use_coordination_service);

  explicit DistributedRuntimeService(
      const DistributedRuntimeServiceImpl::Options& options,
      ::grpc::ServerBuilder* builder, bool use_coordination_service);
  ~DistributedRuntimeService();

  DistributedRuntimeService(const DistributedRuntimeService&) = delete;
  DistributedRuntimeService(DistributedRuntimeService&&) = delete;
  DistributedRuntimeService& operator=(const DistributedRuntimeService&) =
      delete;
  DistributedRuntimeService& operator=(DistributedRuntimeService&&) = delete;

  void Shutdown();

  ::grpc::Server* server() const { return server_.get(); }

 private:
  std::unique_ptr<DistributedRuntimeServiceImpl> impl_;
  std::unique_ptr<CoordinationServiceImpl> coord_impl_;
  std::unique_ptr<::grpc::Server> server_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_PJRT_DISTRIBUTED_SERVICE_H_
