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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_SHUFFLE_AND_REPEAT_FUSION_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_SHUFFLE_AND_REPEAT_FUSION_H_

#include "optimizer_base.h"

namespace tensorflow {
namespace grappler {

class ShuffleAndRepeatFusion : public TFDataOptimizerBase {
 public:
  ShuffleAndRepeatFusion() = default;
  ~ShuffleAndRepeatFusion() override = default;

  string name() const override { return "shuffle_and_repeat_fusion"; };

  bool UsesFunctionLibrary() const override { return false; }

  Status Init(
      const tensorflow::RewriterConfig_CustomGraphOptimizer* config) override {
    return OkStatus();
  }

  Status OptimizeAndCollectStats(Cluster* cluster, const GrapplerItem& item,
                                 GraphDef* output,
                                 OptimizationStats* stats) override;
};

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DATA_SHUFFLE_AND_REPEAT_FUSION_H_
