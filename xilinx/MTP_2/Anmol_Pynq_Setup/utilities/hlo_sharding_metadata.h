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

#ifndef TENSORFLOW_COMPILER_XLA_HLO_IR_HLO_SHARDING_METADATA_H_
#define TENSORFLOW_COMPILER_XLA_HLO_IR_HLO_SHARDING_METADATA_H_

#include <memory>
#include <string>
#include <utility>

#include "flat_hash_map.h"
#include "span.h"
#include "hlo_domain_metadata.h"
#include "hlo_instruction.h"
#include "hlo_sharding.h"
#include "status.h"

namespace xla {

// A DomainMetadata implementation that internally wraps a sharding attribute.
class ShardingMetadata : public DomainMetadata {
 public:
  explicit ShardingMetadata(std::shared_ptr<const HloSharding> sharding)
      : sharding_(std::move(sharding)) {}

  std::unique_ptr<DomainMetadata> Clone() const override;

  absl::string_view Kind() const override { return KindName(); }

  bool Matches(const DomainMetadata& other) const override;

  template <typename H>
  friend H AbslHashValue(H h, const ShardingMetadata& sharding_metadata) {
    const bool has_sharding = sharding_metadata.sharding_ != nullptr;
    if (has_sharding) {
      h = H::combine(std::move(h), *sharding_metadata.sharding_);
    }
    return H::combine(std::move(h), has_sharding);
  }

  size_t Hash() const override { return absl::HashOf(*this); }

  std::string ToString() const override;

  const HloSharding* sharding() const { return sharding_.get(); }

  static absl::string_view KindName() { return "sharding"; }

  static StatusOr<const ShardingMetadata*> ToShardingMetadata(
      const DomainMetadata* metadata);

  // Apply the specified domain metadata onto the specified domain. If no
  // metadata is specified then apply sharding heuristics and normalize the
  // instructions whose sharding deviates from the one which is inferred as to
  // be the original one. Policy wise, HLO passes are allowed to create new
  // unassigned instructions, but if they do create assigned ones, they have to
  // conform to the ones around.
  static Status NormalizeShardingDomain(const DomainMetadata::Domain& domain,
                                        const DomainMetadata* metadata);

 private:
  std::shared_ptr<const HloSharding> sharding_;
};

// If the sharding between root and instruction changes then returns a
// ShardingMetadata based kDomain instruction what can be used to separate
// operand and instruction.
// Returns nullptr if there is no need for a domain separation.
class ShardingDomainCreator {
 public:
  HloInstruction* operator()(HloInstruction* instruction, HloInstruction* root,
                             HloInstruction* operand);

 private:
  // Map from instruction and user sharding to domain users to CSE identical
  // domains.
  struct DomainCseMapKey {
    const HloInstruction* instruction;
    std::shared_ptr<const HloSharding> sharding;

    bool operator==(const DomainCseMapKey& other) const;

    template <typename H>
    friend H AbslHashValue(H h, const DomainCseMapKey& key) {
      h = H::combine(std::move(h), key.instruction);
      const bool has_sharding = key.sharding != nullptr;
      if (has_sharding) {
        h = H::combine(std::move(h), *key.sharding);
      }
      return H::combine(std::move(h), has_sharding);
    }
  };
  absl::flat_hash_map<DomainCseMapKey, HloInstruction*> domain_cse_map_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_HLO_IR_HLO_SHARDING_METADATA_H_
