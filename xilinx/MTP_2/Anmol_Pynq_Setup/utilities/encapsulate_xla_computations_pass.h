/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
// Rewrites computations generated by the xla.compile() Python code into
// XlaLaunch nodes.
//
// xla.compile() does two main things:
// a) marks operators that make up an XLA computation with the attribute
//    _xla_compile_id=XYZ, where XYZ is a unique key.
// b) adds XlaClusterOutput nodes to represent outputs of the computation.
//    These nodes are not marked with the _xla_compile_id attribute.

#ifndef TENSORFLOW_COMPILER_JIT_ENCAPSULATE_XLA_COMPUTATIONS_PASS_H_
#define TENSORFLOW_COMPILER_JIT_ENCAPSULATE_XLA_COMPUTATIONS_PASS_H_

#include <functional>
#include <string>

#include "optimization_registry.h"
#include "graph.h"
#include "env.h"
#include "statusor.h"

namespace tensorflow {

// Encapsulates nodes marked with the _xla_compile_id attribute into
// XlaLaunch operators.
class EncapsulateXlaComputationsPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;

  // The following methods are public only for unit tests.

  // This pass has two stages:
  // a) first, we call EncapsulateSubgraphsPass to encapsulate all nodes
  //    marked with the same _xla_compile_id attribute into functions. These
  //    functions contain the computations to be passed to XlaLaunch. During
  //    encapsulation, we sort the arguments into the order expected by
  //    XlaLaunch.
  static Status Encapsulate(std::unique_ptr<Graph>* graph,
                            FunctionLibraryDefinition* flib_def);

  // b) we rewrite the function calls generated in phase (a) into XlaLaunch
  //    operators. We also convert the XlaClusterOutput output nodes of the
  //    function call into the outputs of the XlaLaunch operator.
  static Status BuildXlaLaunchOps(Graph* graph);

  struct XlaFunctionInfo {
    int variable_start_index = -1;
    std::string function_name;
  };

  // We need to introduce this version to adapt to the output of gpu inference
  // converter. The single argument overload version calls this function.
  //
  // When add_edges_to_output_of_downstream_nodes is true, the output edges of
  // the xla_launch_node's immediate downstream nodes would be attached to the
  // generated xla node. For example, if the original graph is
  // StatefulPartitionedCall{_xla_compile_id=1} -> XlaClusterOutput -> NodeA
  // The output graph of this function would look like the following when
  // add_edges_to_output_of_downstream_nodes is true:
  // XlaLaunch -> NodeA
  static Status BuildXlaLaunchOps(
      Graph* graph,
      const std::function<StatusOr<bool>(const Node&)>& is_xla_launch_node,
      const std::function<StatusOr<XlaFunctionInfo>(const Node&)>&
          get_xla_function_info,
      bool add_edges_to_output_of_downstream_nodes);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_JIT_ENCAPSULATE_XLA_COMPUTATIONS_PASS_H_
