/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SESSION_H_
#define TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SESSION_H_

#include <memory>
#include <string>
#include <vector>

#include "call_options.h"
#include "message_wrappers.h"
#include "grpc_channel.h"
#include "graph.pb.h"
#include "tensor.h"
#include "errors.h"
#include "status.h"
#include "logging.h"
#include "macros.h"
#include "mutex.h"
#include "thread_annotations.h"
#include "config.pb.h"
#include "master.pb.h"
#include "session.h"
#include "session_options.h"

namespace tensorflow {

class MasterInterface;

// A Session instance lets the caller drive a TensorFlow graph
// computation on potentially remote sets of devices. This is a thin
// wrapper around tensorflow::grpc::MasterService.
//
// Multiple threads must synchronize their accesses to a single
// session.
class GrpcSession : public Session {
 protected:
  explicit GrpcSession(const SessionOptions& options);

 public:
  static Status Create(const SessionOptions& options,
                       std::unique_ptr<GrpcSession>* out_session);
  // Resets the resource containers.
  static Status Reset(const SessionOptions& options,
                      const std::vector<string>& containers);

  ~GrpcSession() override;

  // Creates a session with the "target". The session carries out
  // the graph computation defined by "graph", and will have version
  // number "initial_version".
  Status Create(const GraphDef& graph) override;
  Status Create(const RunOptions& run_options, const GraphDef& graph) override;
  Status Create(GraphDef&& graph) override;
  Status Create(const RunOptions& run_options, GraphDef&& graph) override;

  // Runs with and without RunOptions.
  Status Run(const std::vector<std::pair<string, Tensor> >& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs) override;
  Status Run(const RunOptions& run_options,
             const std::vector<std::pair<string, Tensor> >& inputs,
             const std::vector<string>& output_tensor_names,
             const std::vector<string>& target_node_names,
             std::vector<Tensor>* outputs, RunMetadata* run_metadata) override;

  Status Extend(const GraphDef& graph) override;
  Status Extend(const RunOptions& run_options, const GraphDef& graph) override;
  Status Extend(GraphDef&& graph) override;
  Status Extend(const RunOptions& run_options, GraphDef&& graph) override;

  Status Close() override;

  // NOTE: This API is still experimental and may change.
  Status PRunSetup(const std::vector<string>& input_names,
                   const std::vector<string>& output_names,
                   const std::vector<string>& target_nodes,
                   string* handle) override;

  // NOTE: This API is still experimental and may change.
  Status PRun(const string& handle,
              const std::vector<std::pair<string, Tensor> >& inputs,
              const std::vector<string>& output_names,
              std::vector<Tensor>* outputs) override;

  Status ListDevices(std::vector<DeviceAttributes>* response) override;

  Status MakeCallable(const CallableOptions& callable_options,
                      CallableHandle* out_handle) override;
  Status RunCallable(CallableHandle handle,
                     const std::vector<Tensor>& feed_tensors,
                     std::vector<Tensor>* fetch_tensors,
                     RunMetadata* run_metadata) override;
  Status ReleaseCallable(CallableHandle handle) override;

 protected:
  // Takes ownership of `*master`.
  void SetRemoteMaster(std::unique_ptr<MasterInterface> master);
  // Allows subclasses to customize Session creation.
  void SetHandleAndGraphVersion(string handle, int64_t graph_version)
      TF_LOCKS_EXCLUDED(mu_);

 private:
  const SessionOptions options_;
  std::unique_ptr<MasterInterface> master_;
  mutex mu_;

  // handle_ returned by the master to identify this session.
  string handle_ TF_GUARDED_BY(mu_);

  // The current version of the graph.
  int64_t current_graph_version_ TF_GUARDED_BY(mu_);

  bool is_local_ = false;

  Status Handle(string* out_handle) TF_LOCKS_EXCLUDED(mu_);

  Status RunHelper(const RunOptions& run_options,
                   const std::vector<std::pair<string, Tensor> >& inputs,
                   const std::vector<string>& output_tensor_names,
                   const std::vector<string>& target_node_names,
                   std::vector<Tensor>* outputs, RunMetadata* run_metadata,
                   const string& prun_handle);

  Status RunProto(CallOptions* call_options, MutableRunStepRequestWrapper* req,
                  MutableRunStepResponseWrapper* resp);

  // Implementations for all the public interfaces.
  Status CreateImpl(CallOptions* call_options, GraphDef graph);
  Status ExtendImpl(CallOptions* call_options, GraphDef graph);

  TF_DISALLOW_COPY_AND_ASSIGN(GrpcSession);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_DISTRIBUTED_RUNTIME_RPC_GRPC_SESSION_H_
