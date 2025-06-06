/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_JIT_VARIABLE_INFO_UTIL_H_
#define TENSORFLOW_COMPILER_JIT_VARIABLE_INFO_UTIL_H_

#include <map>
#include <optional>
#include <set>
#include <vector>

#include "variable_info.h"
#include "resource_mgr.h"
#include "tensor.h"
#include "status.h"
#include "thread_annotations.h"

namespace tensorflow {

// Snapshot of resource variables for a TF kernel invocation, mapping from
// parameter number to values at execution time. If the resource variable is not
// initialized, the value will not be present.
using ResourceVarsSnapshot = absl::flat_hash_map<int, std::optional<Tensor>>;

// Takes a snapshot of the values of resource variable arguments, whose indices
// are specified in `variable_indices` argument. We snapshot tensors that back
// resource variables since concurrent updates may modify the shape, and it is
// important that the shapes used for compilation match the true shapes of the
// buffers.
//
// We snapshot the entire set of resource variables as one atomic operation.
// This models Read->* dependencies between resource variable operations.  See
// jit/resource_operation_safety_analysis for details.
Status SnapshotResourceVariables(OpKernelContext* ctx,
                                 absl::Span<const int> variable_indices,
                                 absl::Span<VariableInfo const> variable_infos,
                                 ResourceVarsSnapshot* result);

// Acquires the mutexes for all the variables in `variables` using a
// deadlock-safe protocol (acquire the mutexes in increasing-address order).
//
// `variables` is allowed to contain instances that don't track a resource
// variable (i.e. variables[i].var() can be null for some i).
//
// If the variable is read_only(), only acquires reader locks.
Status LockVariables(absl::Span<VariableInfo*> variables)
    TF_EXCLUSIVE_LOCK_FUNCTION();
Status LockVariables(absl::Span<VariableInfo> variables)
    TF_EXCLUSIVE_LOCK_FUNCTION();

// Returns a vector of VariableInfo instances for the resource variable inputs,
// given that *all* inputs are in `inputs`. The input indices for the resource
// variable inputs are in `variable_indices`.
//
// When using the VariableInfos generated by this version, all variables would
// be writer-locked.
Status GetVariableInfosFromInputs(ResourceMgr* rm, DeviceBase* dev,
                                  absl::Span<const Tensor* const> inputs,
                                  absl::Span<const int> variable_indices,
                                  std::vector<VariableInfo>* result);

// variables_updated is a set containing the indices of the variables that are
// going to be mutated. If variables_updated is empty, then in LockVariables all
// variables would only be reader-locked. If variables_updated is null, then we
// consider this information unknown and will acquire writer-lock for all
// variables.
Status GetVariableInfosFromInputs(ResourceMgr* rm, DeviceBase* dev,
                                  absl::Span<const Tensor* const> inputs,
                                  absl::Span<const int> variable_indices,
                                  const std::set<int>* variables_updated,
                                  std::vector<VariableInfo>* result);

std::vector<int> GetResourceVariableIndicesFromContext(OpKernelContext* ctx);

Status CreateVariableInfoLookup(
    absl::Span<VariableInfo const> variable_args,
    absl::flat_hash_map<int, const VariableInfo*>& variable_info_lookup);

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_VARIABLE_INFO_UTIL_H_
