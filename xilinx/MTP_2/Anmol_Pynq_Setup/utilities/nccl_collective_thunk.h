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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_COLLECTIVE_THUNK_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_COLLECTIVE_THUNK_H_

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "function_ref.h"
#include "BuiltinOps.h"  // from @llvm-project
#include "collective_ops_utils.h"
#include "ir_emission_utils.h"
#include "thunk.h"
#include "llvm_util.h"
#include "attribute_exporter.h"
#include "xla_data.pb.h"

#if XLA_ENABLE_XCCL
#include "nccl_utils.h"
#endif  // XLA_ENABLE_XCCL

struct ncclComm;
using ncclComm_t = ncclComm*;

namespace xla {
namespace gpu {

class NcclClique;

struct NcclCollectiveConfig {
  NcclCollectiveConfig();
  NcclCollectiveConfig(NcclCollectiveConfig&&);
  ~NcclCollectiveConfig();

  NcclCollectiveConfig& operator=(NcclCollectiveConfig&&);

  int64_t operand_count;
  std::vector<PrimitiveType> operand_element_type;
  std::vector<ReplicaGroup> replica_groups;
  RendezvousKey::CollectiveOpKind collective_op_kind;
  int64_t op_id;
  CollectiveOpGroupMode group_mode;

  template <typename OpT>
  void SetCollectiveOpKindAndID(OpT op);
  bool IsDegenerate(int64_t replica_count, int64_t partition_count) const;
};

template <typename OpT>
void NcclCollectiveConfig::SetCollectiveOpKindAndID(OpT op) {
  if (op.getChannelId()) {
    collective_op_kind = RendezvousKey::kCrossModule;
    op_id = static_cast<int64_t>(op.getChannelId()->getHandle());
  } else {
    collective_op_kind = RendezvousKey::kCrossReplica;
    mlir::ModuleOp parent = op->template getParentOfType<mlir::ModuleOp>();
    mlir::IntegerAttr unique_id =
        parent->getAttrOfType<mlir::IntegerAttr>("hlo.unique_id");
    op_id = static_cast<int64_t>(unique_id.getInt());
  }
}

template <typename OpT>
NcclCollectiveConfig GetNcclCollectiveConfigForMlir(
    OpT op, std::optional<bool> use_global_device_ids) {
  NcclCollectiveConfig config;
  config.operand_count = op.getInputs().size();
  config.operand_element_type.reserve(config.operand_count);
  for (int i = 0; i < config.operand_count; i++) {
    const Shape shape = GetShape(op.getInputs()[i]);
    config.operand_element_type.push_back(shape.element_type());
  }
  config.replica_groups = ConvertReplicaGroups(op.getReplicaGroups()).value();
  config.SetCollectiveOpKindAndID(op);
  config.group_mode = GetCollectiveOpGroupMode(op.getChannelId().has_value(),
                                               use_global_device_ids)
                          .value();
  return config;
}

// Thunk base class for NCCL collective operations.
class NcclCollectiveThunk : public Thunk {
 public:
  NcclCollectiveThunk(Kind kind, ThunkInfo thunk_info, bool is_sync);

  struct Buffer {
    int64_t element_count;
    BufferAllocation::Slice source_buffer;
    BufferAllocation::Slice destination_buffer;
    mlir::Value source_value;
    mlir::Value destination_value;
  };

  class AsyncExecutor {
   public:
    // Executes the function on the async communications stream and records a
    // completion event.
    Status Execute(
        absl::FunctionRef<Status(const ExecuteParams&, se::Stream&, ncclComm_t)>
            fn,
        const ExecuteParams& params, ncclComm_t comm,
        AsyncStreamKind stream_kind);
    // Blocks the compute stream until async communication is complete.
    Status Await(const ExecuteParams& params);

   private:
    absl::Mutex mu_;
    // Store done events (by device ordinal) for the done thunk to wait on.
    absl::flat_hash_map<int, se::Event> done_events_ ABSL_GUARDED_BY(mu_);
  };

  // Returns whether NCCL operations appear possible to perform; e.g. if we
  // haven't done a build with the CUDA compiler enabled, we can't compile the
  // NCCL header, and thus this will be false.
  //
  // When this is false, the ExecuteOnStream() call will simply return a status
  // error.
  static bool NcclIsEnabled();
  static Status CheckImplementable();

  // Logging support.
  static std::string GetDeviceString(const NcclExecuteParams& params);

  AsyncExecutor* async_executor() { return async_.get(); }
  Status ExecuteOnStream(const ExecuteParams& params) override;

 protected:
  virtual Status RunNcclCollective(const ExecuteParams& params,
                                   se::Stream& stream, ncclComm_t comm) = 0;
  virtual const NcclCollectiveConfig& config() const = 0;
  virtual AsyncStreamKind GetAsyncStreamKind() const {
    return kAsyncStreamCollective;
  }

 private:
  bool IsAsync() const { return async_ != nullptr; }
  int64_t GetStreamId() const {
    return IsAsync() ? 1 + GetAsyncStreamKind() : 0;
  }

#if XLA_ENABLE_XCCL
  bool first_call_to_execute_ = true;
#endif  // XLA_ENABLE_XCCL
  std::unique_ptr<AsyncExecutor> async_;  // null if not async.
};

class NcclCollectiveDoneThunk : public Thunk {
 public:
  NcclCollectiveDoneThunk(Thunk::Kind kind, ThunkInfo thunk_info,
                          NcclCollectiveThunk::AsyncExecutor& async);

  Status ExecuteOnStream(const ExecuteParams& params) override;

 private:
  NcclCollectiveThunk::AsyncExecutor& async_;
};

Status IsValidOperand(mlir::Value operand, Thunk::Kind reduction_op);

template <typename NcclThunkType, typename OpT>
Status AddOpDescription(Status status, OpT op, int64_t replica_count,
                        int64_t partition_count) {
  if (status.ok()) {
    return status;
  }
  CollectiveOpGroupMode group_mode = NcclThunkType::GetGroupMode(op);
  return Status(
      status.code(),
      absl::StrFormat(
          "%s\n"
          "%s with replica_count: %d, partition_count: %d, group_mode: %s, "
          "operand_count: %d\n%s",
          status.message(), NcclThunkType::GetHloOpName(), replica_count,
          partition_count, CollectiveOpGroupModeToString(group_mode),
          op->getNumOperands() / 2, llvm_ir::DumpToString(op.getOperation())));
}

#if XLA_ENABLE_XCCL
// TODO(hanbinyoon): Consider moving to nccl_utils.h when deprecating Thunks.
StatusOr<NcclComm::Lock> LockNcclComm(
    const NcclExecuteParams& params,
    const std::vector<ReplicaGroup>& replica_groups,
    CollectiveOpGroupMode group_mode, int64_t op_id, int64_t stream_id);
#endif  // XLA_ENABLE_XCCL

struct DeviceBufferPair {
  PrimitiveType element_type;
  int64_t element_count;
  se::DeviceMemoryBase source_buffer;
  se::DeviceMemoryBase destination_buffer;
};
StatusOr<std::vector<DeviceBufferPair>> ConvertToDeviceBuffers(
    const Thunk::ExecuteParams& params,
    const std::vector<NcclCollectiveThunk::Buffer>& buffers,
    const std::vector<PrimitiveType>& element_types);

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_NCCL_COLLECTIVE_THUNK_H_
