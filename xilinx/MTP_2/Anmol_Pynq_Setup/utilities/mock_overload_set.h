//
// Copyright 2019 The Abseil Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ABSL_RANDOM_INTERNAL_MOCK_OVERLOAD_SET_H_
#define ABSL_RANDOM_INTERNAL_MOCK_OVERLOAD_SET_H_

#include <type_traits>

#include "gmock.h"
#include "mock_helpers.h"
#include "mocking_bit_gen.h"

namespace absl {
ABSL_NAMESPACE_BEGIN
namespace random_internal {

template <typename DistrT, typename Fn>
struct MockSingleOverload;

// MockSingleOverload
//
// MockSingleOverload hooks in to gMock's `ON_CALL` and `EXPECT_CALL` macros.
// EXPECT_CALL(mock_single_overload, Call(...))` will expand to a call to
// `mock_single_overload.gmock_Call(...)`. Because expectations are stored on
// the MockingBitGen (an argument passed inside `Call(...)`), this forwards to
// arguments to MockingBitGen::Register.
//
// The underlying KeyT must match the KeyT constructed by DistributionCaller.
template <typename DistrT, typename Ret, typename... Args>
struct MockSingleOverload<DistrT, Ret(MockingBitGen&, Args...)> {
  static_assert(std::is_same<typename DistrT::result_type, Ret>::value,
                "Overload signature must have return type matching the "
                "distribution result_type.");
  using KeyT = Ret(DistrT, std::tuple<Args...>);

  template <typename MockURBG>
  auto gmock_Call(MockURBG& gen, const ::testing::Matcher<Args>&... matchers)
      -> decltype(MockHelpers::MockFor<KeyT>(gen).gmock_Call(matchers...)) {
    static_assert(std::is_base_of<MockingBitGen, MockURBG>::value,
                  "Mocking requires an absl::MockingBitGen");
    return MockHelpers::MockFor<KeyT>(gen).gmock_Call(matchers...);
  }
};

template <typename DistrT, typename Ret, typename Arg, typename... Args>
struct MockSingleOverload<DistrT, Ret(Arg, MockingBitGen&, Args...)> {
  static_assert(std::is_same<typename DistrT::result_type, Ret>::value,
                "Overload signature must have return type matching the "
                "distribution result_type.");
  using KeyT = Ret(DistrT, std::tuple<Arg, Args...>);

  template <typename MockURBG>
  auto gmock_Call(const ::testing::Matcher<Arg>& matcher, MockURBG& gen,
                  const ::testing::Matcher<Args>&... matchers)
      -> decltype(MockHelpers::MockFor<KeyT>(gen).gmock_Call(matcher,
                                                             matchers...)) {
    static_assert(std::is_base_of<MockingBitGen, MockURBG>::value,
                  "Mocking requires an absl::MockingBitGen");
    return MockHelpers::MockFor<KeyT>(gen).gmock_Call(matcher, matchers...);
  }
};

// MockOverloadSet
//
// MockOverloadSet takes a distribution and a collection of signatures and
// performs overload resolution amongst all the overloads. This makes
// `EXPECT_CALL(mock_overload_set, Call(...))` expand and do overload resolution
// correctly.
template <typename DistrT, typename... Signatures>
struct MockOverloadSet;

template <typename DistrT, typename Sig>
struct MockOverloadSet<DistrT, Sig> : public MockSingleOverload<DistrT, Sig> {
  using MockSingleOverload<DistrT, Sig>::gmock_Call;
};

template <typename DistrT, typename FirstSig, typename... Rest>
struct MockOverloadSet<DistrT, FirstSig, Rest...>
    : public MockSingleOverload<DistrT, FirstSig>,
      public MockOverloadSet<DistrT, Rest...> {
  using MockSingleOverload<DistrT, FirstSig>::gmock_Call;
  using MockOverloadSet<DistrT, Rest...>::gmock_Call;
};

}  // namespace random_internal
ABSL_NAMESPACE_END
}  // namespace absl
#endif  // ABSL_RANDOM_INTERNAL_MOCK_OVERLOAD_SET_H_
