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

#ifndef TENSORFLOW_TSL_PROFILER_UTILS_PREPROCESS_XPLANE_H_
#define TENSORFLOW_TSL_PROFILER_UTILS_PREPROCESS_XPLANE_H_

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "container.h"
#include "memory.h"
#include "string_view.h"
#include "optional.h"
#include "context_types.h"
#include "xplane.pb.h"
#include "tpu_xplane_utils.h"
#include "trace_utils.h"
#include "xplane_builder.h"
#include "xplane_schema.h"

namespace tsl {
namespace profiler {

static constexpr uint32_t kRunIdMask = (1U << 27) - 1;

/*
 * Subclass of this interface will perform different mutatation to the event.
 * Checking eligibilities of event mutation is not responsible of this class.
 */
class XplaneEventMutator {
 public:
  virtual ~XplaneEventMutator() = default;

  // Mutate event by event specifiedd by the event_metadata.
  virtual void Mutate(XEventBuilder* builder) = 0;
  // Mutate line by line if event_metadata() return nullptr.
  virtual void MutateEventsInLine(XLineBuilder* line) = 0;

  const XEventMetadata* event_metadata() const { return event_metadata_; }

 protected:
  explicit XplaneEventMutator(XEventMetadata* event_metadata)
      : event_metadata_(event_metadata) {}

  XEventMetadata* event_metadata_;
};

class XplaneEventMutatorFactory {
 public:
  virtual ~XplaneEventMutatorFactory() = default;

  virtual std::vector<std::unique_ptr<XplaneEventMutator>> CreateMutators(
      XPlaneBuilder* xplane) const = 0;

 protected:
  XplaneEventMutatorFactory() = default;
};

/*
 * mutate specific HostEventType by adding "_r" Xstats, which equal to the
 * specified root level.
 */
class XplaneRootEventMutatorFactory : public XplaneEventMutatorFactory {
 public:
  static std::unique_ptr<XplaneEventMutatorFactory> CreateFactory(
      HostEventType event_type, int64_t root_level) {
    std::unique_ptr<XplaneEventMutatorFactory> base;
    base.reset(new XplaneRootEventMutatorFactory(event_type, root_level));
    return base;
  }

  std::vector<std::unique_ptr<XplaneEventMutator>> CreateMutators(
      XPlaneBuilder* xplane) const override {
    std::vector<std::unique_ptr<XplaneEventMutator>> mutators;
    XEventMetadata* event_metadata =
        xplane->GetEventMetadata(GetHostEventTypeStr(event_type_));
    if (event_metadata == nullptr) return {};
    XStatMetadata* root_metadata =
        xplane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kIsRoot));
    mutators.emplace_back(std::make_unique<XplaneRootEventMutator>(
        event_metadata, root_metadata, root_level_));
    return mutators;
  }

 private:
  explicit XplaneRootEventMutatorFactory(HostEventType event_type,
                                         int64_t root_level)
      : event_type_(event_type), root_level_(root_level) {}

  class XplaneRootEventMutator : public XplaneEventMutator {
   public:
    XplaneRootEventMutator(XEventMetadata* event_metadata,
                           XStatMetadata* root_stats_metadata,
                           int64_t root_level)
        : XplaneEventMutator(event_metadata),
          root_stats_metadata_(root_stats_metadata),
          root_level_(root_level) {}
    void Mutate(XEventBuilder* event_builder) override {
      event_builder->SetOrAddStatValue(*root_stats_metadata_, root_level_);
    }
    void MutateEventsInLine(XLineBuilder* line) override {
      CHECK(false);  // Crash OK
    }

   private:
    XStatMetadata* root_stats_metadata_;
    int64_t root_level_;
  };

  HostEventType event_type_;
  int64_t root_level_;
};

template <typename StatValueType, StatType kStatId>
class XContextStatsAccessor {
 public:
  using value_type = StatValueType;

  bool Initialize(XPlaneBuilder* xplane) {
    stats_metadata_ = xplane->GetStatMetadata(GetStatTypeStr(kStatId));
    return stats_metadata_;
  }

  std::optional<StatValueType> GetStat(XEventBuilder* event_builder) {
    auto* stat = event_builder->GetStat(*stats_metadata_);
    if (stat == nullptr) return std::nullopt;
    if constexpr (std::is_integral_v<StatValueType>) {
      return event_builder->IntOrUintValue(*stat);
    } else {
      return event_builder->StrOrRefValue(*stat);
    }
  }

 private:
  XStatMetadata* stats_metadata_ = nullptr;
};

// A template helper for tuple manipulation, although std::apply can achieve
// similar result. However it requires C++ 17, TF windows bot is still C++ 14.
template <std::size_t... Idx>
auto make_index_dispatcher(std::index_sequence<Idx...>) {
  return [](auto&& f) { (f(std::integral_constant<std::size_t, Idx>{}), ...); };
}

template <std::size_t N>
auto make_index_dispatcher() {
  return make_index_dispatcher(std::make_index_sequence<N>{});
}

template <typename Tuple, typename Func>
void for_each(Tuple&& t, Func&& f) {
  constexpr auto n = std::tuple_size<std::decay_t<Tuple>>::value;
  auto dispatcher = make_index_dispatcher<n>();
  dispatcher([&f, &t](auto idx) { f(std::get<idx>(std::forward<Tuple>(t))); });
}

/*
 * mutate specific pair of HostEventType with specified XStats list by adding
 * relevant producer and consumer connected TraceMe 2.0 semantics.
 * 1. both produer and consumer side of smenatics is populated,
 * 2. using the specified ContextType.
 * 3. unique context id is automatically generated.
 *    if the combination of stats value is unique under specified context_type,
 *    then set unique_stats true, then context_id is a hash of stats tuple.
 *    otherwise (unique_stats = false), context id is computed as a hash of
 *    tuple <producer_event, consumer_event, stats>
 */
template <HostEventType producer_event, HostEventType consumer_event,
          ContextType context_type, bool unique_stats,
          typename... StatsAccessorTypes>
class XplaneConnectedEventMutatorFactory : public XplaneEventMutatorFactory {
 public:
  static std::unique_ptr<XplaneEventMutatorFactory> CreateFactory() {
    return absl::WrapUnique(new XplaneConnectedEventMutatorFactory());
  }

  using StatsAccessors = std::tuple<StatsAccessorTypes...>;

  std::vector<std::unique_ptr<XplaneEventMutator>> CreateMutators(
      XPlaneBuilder* xplane) const override {
    // Check if all stats exist in current plane.
    StatsAccessors stats_accessors;
    bool all_required_stats_exist = true;
    auto check_stats_meta = [&all_required_stats_exist,
                             xplane](auto&& accessor) {
      if (all_required_stats_exist == false) return;
      if (!accessor.Initialize(xplane)) all_required_stats_exist = false;
    };
    for_each(stats_accessors, check_stats_meta);
    if (!all_required_stats_exist) return {};

    XEventMetadata* producer_event_metadata =
        xplane->GetEventMetadata(GetHostEventTypeStr(producer_event));
    XEventMetadata* consumer_event_metadata =
        xplane->GetEventMetadata(GetHostEventTypeStr(consumer_event));

    std::vector<std::unique_ptr<XplaneEventMutator>> mutators;
    if (producer_event_metadata) {
      XStatMetadata* context_type_metadata = xplane->GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kProducerType));
      XStatMetadata* context_id_metadata = xplane->GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kProducerId));
      mutators.emplace_back(std::make_unique<XplaneConnectedEventMutator>(
          producer_event_metadata, context_type_metadata, context_id_metadata,
          stats_accessors));
    }
    if (consumer_event_metadata) {
      XStatMetadata* context_type_metadata = xplane->GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kConsumerType));
      XStatMetadata* context_id_metadata = xplane->GetOrCreateStatMetadata(
          GetStatTypeStr(StatType::kConsumerId));
      mutators.emplace_back(std::make_unique<XplaneConnectedEventMutator>(
          consumer_event_metadata, context_type_metadata, context_id_metadata,
          stats_accessors));
    }
    return mutators;
  }

 private:
  XplaneConnectedEventMutatorFactory() = default;

  class XplaneConnectedEventMutator : public XplaneEventMutator {
   public:
    XplaneConnectedEventMutator(XEventMetadata* event_metadata,
                                XStatMetadata* context_type_metadata,
                                XStatMetadata* context_id_metadata,
                                const StatsAccessors& accessors)
        : XplaneEventMutator(event_metadata),
          context_type_metadata_(context_type_metadata),
          context_id_metadata_(context_id_metadata),
          accessors_(accessors) {}

    void Mutate(XEventBuilder* event_builder) override {
      bool all_required_stats_exist = true;
      std::vector<std::variant<absl::string_view, uint64_t>> required_stats;
      auto check_stats_meta = [&all_required_stats_exist, &required_stats,
                               event_builder](auto&& accessor) {
        if (all_required_stats_exist == false) return;
        auto stats_data = accessor.GetStat(event_builder);
        if (!stats_data) {
          all_required_stats_exist = false;
        } else {
          required_stats.emplace_back(*stats_data);
        }
      };
      for_each(accessors_, check_stats_meta);

      if (!all_required_stats_exist) return;

      int64_t context_id;
      if constexpr (unique_stats) {
        context_id = absl::HashOf(required_stats);
      } else {
        context_id =
            absl::HashOf(producer_event, consumer_event, required_stats);
      }
      event_builder->SetOrAddStatValue(*context_type_metadata_,
                                       static_cast<int64_t>(context_type));
      event_builder->SetOrAddStatValue(*context_id_metadata_, context_id);
    }

    void MutateEventsInLine(XLineBuilder* line) override {
      CHECK(false);  // Crash OK
    }

   private:
    XStatMetadata* context_type_metadata_;
    XStatMetadata* context_id_metadata_;
    StatsAccessors accessors_;
  };
};

template <HostEventType event_type>
class HostRunIdMutatorFactory : public XplaneEventMutatorFactory {
 public:
  static std::unique_ptr<XplaneEventMutatorFactory> CreateFactory() {
    return absl::WrapUnique(new HostRunIdMutatorFactory());
  }

  std::vector<std::unique_ptr<XplaneEventMutator>> CreateMutators(
      XPlaneBuilder* xplane) const override {
    std::vector<std::unique_ptr<XplaneEventMutator>> mutators;
    XEventMetadata* event_metadata =
        xplane->GetEventMetadata(GetHostEventTypeStr(event_type));
    if (event_metadata == nullptr) return {};
    XContextStatsAccessor<int64_t, StatType::kRunId> run_id_stats_accessor;
    run_id_stats_accessor.Initialize(xplane);
    XStatMetadata* run_id_metadata =
        xplane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kRunId));
    mutators.emplace_back(std::make_unique<HostRunIdMutator>(
        event_metadata, run_id_stats_accessor, run_id_metadata));
    return mutators;
  }

 private:
  HostRunIdMutatorFactory() = default;
  class HostRunIdMutator : public XplaneEventMutator {
   public:
    HostRunIdMutator(
        XEventMetadata* event_metadata,
        XContextStatsAccessor<int64_t, StatType::kRunId> run_id_stats_accessor,
        XStatMetadata* run_id_metadata)
        : XplaneEventMutator(event_metadata),
          run_id_stats_accessor_(run_id_stats_accessor),
          run_id_metadata_(run_id_metadata) {}

    void Mutate(XEventBuilder* event_builder) override {
      auto run_id = run_id_stats_accessor_.GetStat(event_builder);
      if (!run_id) return;
      int64_t fixed_run_id = ((uint64_t)run_id.value() & kRunIdMask);
      event_builder->SetOrAddStatValue(*run_id_metadata_, fixed_run_id);
    }

    void MutateEventsInLine(XLineBuilder* line) override {
      CHECK(false);  // Crash OK
    }

   private:
    XContextStatsAccessor<int64_t, StatType::kRunId> run_id_stats_accessor_;
    XStatMetadata* run_id_metadata_;
  };
};

// Line mutator for TPU XLA module line.
// To connect these events with launch events from CPU plane, we need to
// create appropriate TraceMe 2.0 semantics (_c, _ct stats) from their
// device_ordinal(from plane name) / run_id / queue_id stats (from event stats).
class TpuModuleLineMutatorFactory : public XplaneEventMutatorFactory {
 public:
  static std::unique_ptr<XplaneEventMutatorFactory> CreateFactory() {
    return absl::WrapUnique(new TpuModuleLineMutatorFactory());
  }

  std::vector<std::unique_ptr<XplaneEventMutator>> CreateMutators(
      XPlaneBuilder* xplane) const override {
    std::vector<std::unique_ptr<XplaneEventMutator>> mutators;
    if (absl::StartsWith(xplane->Name(), kTpuPlanePrefix) &&
        GetTensorCoreId(xplane->Name()).has_value()) {
      if (auto device_ordinal = ParseDeviceOrdinal(xplane->Name())) {
        XStatMetadata* context_type_metadata = xplane->GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kConsumerType));
        XStatMetadata* context_id_metadata = xplane->GetOrCreateStatMetadata(
            GetStatTypeStr(StatType::kConsumerId));
        XContextStatsAccessor<uint64_t, StatType::kQueueId>
            queue_id_stats_accessor;
        XContextStatsAccessor<uint64_t, StatType::kRunId> run_id_stats_accessor;
        queue_id_stats_accessor.Initialize(xplane);
        run_id_stats_accessor.Initialize(xplane);
        mutators.emplace_back(std::make_unique<TpuModuleLineMutator>(
            *device_ordinal, context_type_metadata, context_id_metadata,
            queue_id_stats_accessor, run_id_stats_accessor));
      }
    }
    return mutators;
  }

 private:
  TpuModuleLineMutatorFactory() = default;

  class TpuModuleLineMutator : public XplaneEventMutator {
   public:
    TpuModuleLineMutator(
        uint32_t device_ordinal, XStatMetadata* context_type_metadata,
        XStatMetadata* context_id_metadata,
        XContextStatsAccessor<uint64_t, StatType::kQueueId>
            queue_id_stats_accessor,
        XContextStatsAccessor<uint64_t, StatType::kRunId> run_id_stats_accessor)
        : XplaneEventMutator(nullptr),
          device_ordinal_(device_ordinal),
          context_type_metadata_(context_type_metadata),
          context_id_metadata_(context_id_metadata),
          queue_id_stats_accessor_(queue_id_stats_accessor),
          run_id_stats_accessor_(run_id_stats_accessor) {}

    void Mutate(XEventBuilder* event_builder) override {
      CHECK(false);  // Crash OK
    }

    void MutateEventsInLine(XLineBuilder* line) override {
      if (line->Name() != kXlaModuleLineName) return;
      line->ForEachEvent([&](XEventBuilder event) {
        auto run_id = run_id_stats_accessor_.GetStat(&event);
        auto queue_id = queue_id_stats_accessor_.GetStat(&event);
        if (!run_id || !queue_id) return;
        // The order of tuple <device_ordinal, queue_id, run_id> need to be
        // consistent with other kTpuLaunch types.
        std::vector<std::variant<absl::string_view, uint64_t>> required_stats;
        required_stats.reserve(3);
        required_stats.emplace_back(device_ordinal_);
        required_stats.emplace_back(*queue_id);
        required_stats.emplace_back(*run_id);
        int64_t context_id = absl::HashOf(required_stats);
        event.SetOrAddStatValue(*context_type_metadata_,
                                static_cast<int64_t>(ContextType::kTpuLaunch));
        event.SetOrAddStatValue(*context_id_metadata_, context_id);
      });
    }

   private:
    uint64_t device_ordinal_;
    XStatMetadata* context_type_metadata_;
    XStatMetadata* context_id_metadata_;
    XContextStatsAccessor<uint64_t, StatType::kQueueId>
        queue_id_stats_accessor_;
    XContextStatsAccessor<uint64_t, StatType::kRunId> run_id_stats_accessor_;
  };
};

// Preprocess the given XSpace to support legacy traces. It converts old context
// events and stats into new ones according to go/xprof-traceme2-semantics.
void PreprocessXSpace(XSpace* space);
void PreprocessXPlane(XPlane* plane);

}  // namespace profiler
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PROFILER_UTILS_PREPROCESS_XPLANE_H_
