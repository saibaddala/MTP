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

// Rewrites computations generated by the tpu.replicate() Python code into
// TPUReplicate operators.
//
// The tpu.replicate() does two main things:
// a) marks operators that make up a TPU computation with the attribute
//    _tpu_replicate=XYZ, where XYZ is a unique key.
// b) adds TPUReplicatedInput and TPUReplicatedOutput nodes to represent
//    replicated inputs. These nodes are not marked with the _tpu_replicate
//    attribute.

#ifndef TENSORFLOW_CORE_TPU_GRAPH_REWRITE_ENCAPSULATE_TPU_COMPUTATIONS_PASS_H_
#define TENSORFLOW_CORE_TPU_GRAPH_REWRITE_ENCAPSULATE_TPU_COMPUTATIONS_PASS_H_

#include "encapsulate_util.h"
#include "optimization_registry.h"
#include "graph.h"

namespace tensorflow {

// Encapsulates nodes marked with the _tpu_replicate attribute into
// TPUReplicate operators.
class EncapsulateTPUComputationsPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;

  // The following methods are public only for unit tests.

  // This pass has two stages:
  // a) first, we call the EncapsulateSubgraphsPass to encapsulate all nodes
  //    marked with the same _tpu_replicate attribute into functions. These
  //    functions contain the computations to be passed to TPUReplicate. During
  //    encapsulation, we sort the arguments into the order expected by
  //    TPUReplicate.
  static Status Encapsulate(std::unique_ptr<Graph>* graph,
                            FunctionLibraryDefinition* flib_def);

  // b) we rewrite the function calls generated in phase (a) into TPUReplicate
  //    operators. We also flatten the TPUReplicatedInput and
  //    TPUReplicatedOutput replicated input and output nodes of the function
  //    call into the replicated input and outputs of the TPUReplicate operator.
  static Status BuildTPUReplicateOps(Graph* graph);
};

// Graph optimization pass that calls `ExtractOutsideCompilation` for all XLA
// computation nodes.
class ExtractOutsideCompilationPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;

  static Status ProcessHeadTailOutsideCompilation(
      const string& outside_compilation_attr_name, int* lifted_arg_count,
      std::unordered_map<string, XlaClusterInfo>* clusters, Graph* g,
      FunctionLibraryRuntime* flr, FunctionLibraryDefinition* fld);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_GRAPH_REWRITE_ENCAPSULATE_TPU_COMPUTATIONS_PASS_H_
