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
#ifndef TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_CONVERT_GRAPH_H_
#define TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_CONVERT_GRAPH_H_

#include <vector>

#include "convert_nodes.h"
#include "trt_optimization_pass.h"
#include "utils.h"
#include "trt_shape_optimization_profiles.h"
#include "graph.pb.h"
#include "tensor_shape.h"
#include "cluster.h"
#include "grappler_item.h"
#include "status.h"
#include "types.h"

#if GOOGLE_CUDA && GOOGLE_TENSORRT

namespace tensorflow {
namespace tensorrt {
namespace convert {

// These functions are internal implementation functions for the
// TRTOptimizationPass.

// Performs segmentation and conversion on the given Grappler item. This method
// contains the core logic of the TRTOptimizationPass.
Status ConvertGraph(const TRTOptimizationPass::ConversionParams& params,
                    grappler::GrapplerItem& grappler_item,
                    const std::vector<string>& input_output_names,
                    grappler::Cluster* cluster, GraphDef* output);

// Helper method for the conversion, expose for testing.
std::pair<int, Allocator*> GetDeviceAndAllocator(
    const grappler::Cluster* cluster, const EngineInfo& engine);

// Helper method that registers `segment_graph` as a function to the function
// library in `graph`.
Status RegisterGraphToFunctionLibrary(const GraphDef& segment_graph_def,
                                      Graph* graph, const string& engine_name);

// Creates and serializes an ICudaEngine. Used only in is_dynamic_op=false,
// a.k.a. static engine mode.
Status CreateStaticEngine(const TRTOptimizationPass::ConversionParams& params,
                          const EngineInfo& info, int max_batch_size,
                          const std::vector<PartialTensorShape>& input_shapes,
                          TrtShapeOptimizationProfile* profile,
                          string* segment_string, grappler::Cluster* cluster);

}  // namespace convert
}  // namespace tensorrt
}  // namespace tensorflow

#endif  // GOOGLE_CUDA && GOOGLE_TENSORRT

#endif  // TENSORFLOW_COMPILER_TF2TENSORRT_CONVERT_CONVERT_GRAPH_H_
