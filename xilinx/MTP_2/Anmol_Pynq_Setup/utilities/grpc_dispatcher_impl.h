/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DATA_SERVICE_GRPC_DISPATCHER_IMPL_H_
#define TENSORFLOW_CORE_DATA_SERVICE_GRPC_DISPATCHER_IMPL_H_

#include "server_builder.h"
#include "dispatcher.grpc.pb.h"
#include "dispatcher_impl.h"
#include "export.pb.h"
#include "service_config.pb.h"

namespace tensorflow {
namespace data {

// This class is a wrapper that handles communication for gRPC.
class GrpcDispatcherImpl : public DispatcherService::Service {
 public:
  // Constructs a GrpcDispatcherImpl with the given config, and registers it
  // with `server_builder`.
  explicit GrpcDispatcherImpl(const experimental::DispatcherConfig& config,
                              ::grpc::ServerBuilder& server_builder);
  ~GrpcDispatcherImpl() override {}

  Status Start();

  size_t NumActiveIterations();

  DispatcherStateExport ExportState() const;

#define HANDLER(method)                                 \
  ::grpc::Status method(::grpc::ServerContext* context, \
                        const method##Request* request, \
                        method##Response* response) override;
  HANDLER(WorkerHeartbeat);
  HANDLER(WorkerUpdate);
  HANDLER(GetDatasetDef);
  HANDLER(GetSplit);
  HANDLER(GetVersion);
  HANDLER(GetOrRegisterDataset);
  HANDLER(ReleaseIterationClient);
  HANDLER(MaybeRemoveTask);
  HANDLER(GetOrCreateJob);
  HANDLER(GetOrCreateIteration);
  HANDLER(ClientHeartbeat);
  HANDLER(GetWorkers);
  HANDLER(GetDataServiceMetadata);
  HANDLER(GetDataServiceConfig);
  HANDLER(Snapshot);
  HANDLER(GetSnapshotSplit);
  HANDLER(GetSnapshotStreams);
#undef HANDLER

 private:
  DataServiceDispatcherImpl impl_;

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcDispatcherImpl);
};

}  // namespace data
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DATA_SERVICE_GRPC_DISPATCHER_IMPL_H_
