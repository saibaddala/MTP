// Copyright 2022 The Abseil Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef ABSL_LOG_INTERNAL_LOG_IMPL_H_
#define ABSL_LOG_INTERNAL_LOG_IMPL_H_

#include "conditions.h"
#include "log_message.h"
#include "strip.h"

// ABSL_LOG()
#define ABSL_LOG_IMPL(severity)                          \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATELESS, true) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

// ABSL_PLOG()
#define ABSL_PLOG_IMPL(severity)                           \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATELESS, true)   \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream() \
          .WithPerror()

// ABSL_DLOG()
#ifndef NDEBUG
#define ABSL_DLOG_IMPL(severity)                         \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATELESS, true) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()
#else
#define ABSL_DLOG_IMPL(severity)                          \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATELESS, false) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()
#endif

#define ABSL_LOG_IF_IMPL(severity, condition)                 \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATELESS, condition) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()
#define ABSL_PLOG_IF_IMPL(severity, condition)                \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATELESS, condition) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()    \
          .WithPerror()

#ifndef NDEBUG
#define ABSL_DLOG_IF_IMPL(severity, condition)                \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATELESS, condition) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()
#else
#define ABSL_DLOG_IF_IMPL(severity, condition)                           \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATELESS, false && (condition)) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()
#endif

// ABSL_LOG_EVERY_N
#define ABSL_LOG_EVERY_N_IMPL(severity, n)                         \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, true)(EveryN, n) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

// ABSL_LOG_FIRST_N
#define ABSL_LOG_FIRST_N_IMPL(severity, n)                         \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, true)(FirstN, n) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

// ABSL_LOG_EVERY_POW_2
#define ABSL_LOG_EVERY_POW_2_IMPL(severity)                        \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, true)(EveryPow2) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

// ABSL_LOG_EVERY_N_SEC
#define ABSL_LOG_EVERY_N_SEC_IMPL(severity, n_seconds)                        \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, true)(EveryNSec, n_seconds) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_PLOG_EVERY_N_IMPL(severity, n)                        \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, true)(EveryN, n) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()         \
          .WithPerror()

#define ABSL_PLOG_FIRST_N_IMPL(severity, n)                        \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, true)(FirstN, n) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()         \
          .WithPerror()

#define ABSL_PLOG_EVERY_POW_2_IMPL(severity)                       \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, true)(EveryPow2) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()         \
          .WithPerror()

#define ABSL_PLOG_EVERY_N_SEC_IMPL(severity, n_seconds)                       \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, true)(EveryNSec, n_seconds) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()                    \
          .WithPerror()

#ifndef NDEBUG
#define ABSL_DLOG_EVERY_N_IMPL(severity, n)        \
  ABSL_LOG_INTERNAL_CONDITION_INFO(STATEFUL, true) \
  (EveryN, n) ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_DLOG_FIRST_N_IMPL(severity, n)        \
  ABSL_LOG_INTERNAL_CONDITION_INFO(STATEFUL, true) \
  (FirstN, n) ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_DLOG_EVERY_POW_2_IMPL(severity)       \
  ABSL_LOG_INTERNAL_CONDITION_INFO(STATEFUL, true) \
  (EveryPow2) ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_DLOG_EVERY_N_SEC_IMPL(severity, n_seconds) \
  ABSL_LOG_INTERNAL_CONDITION_INFO(STATEFUL, true)      \
  (EveryNSec, n_seconds) ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#else  // def NDEBUG
#define ABSL_DLOG_EVERY_N_IMPL(severity, n)         \
  ABSL_LOG_INTERNAL_CONDITION_INFO(STATEFUL, false) \
  (EveryN, n) ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_DLOG_FIRST_N_IMPL(severity, n)         \
  ABSL_LOG_INTERNAL_CONDITION_INFO(STATEFUL, false) \
  (FirstN, n) ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_DLOG_EVERY_POW_2_IMPL(severity)        \
  ABSL_LOG_INTERNAL_CONDITION_INFO(STATEFUL, false) \
  (EveryPow2) ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_DLOG_EVERY_N_SEC_IMPL(severity, n_seconds) \
  ABSL_LOG_INTERNAL_CONDITION_INFO(STATEFUL, false)     \
  (EveryNSec, n_seconds) ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()
#endif  // def NDEBUG

#define ABSL_LOG_IF_EVERY_N_IMPL(severity, condition, n)                \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, condition)(EveryN, n) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_LOG_IF_FIRST_N_IMPL(severity, condition, n)                \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, condition)(FirstN, n) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_LOG_IF_EVERY_POW_2_IMPL(severity, condition)               \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, condition)(EveryPow2) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_LOG_IF_EVERY_N_SEC_IMPL(severity, condition, n_seconds)    \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, condition)(EveryNSec, \
                                                             n_seconds) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_PLOG_IF_EVERY_N_IMPL(severity, condition, n)               \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, condition)(EveryN, n) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()              \
          .WithPerror()

#define ABSL_PLOG_IF_FIRST_N_IMPL(severity, condition, n)               \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, condition)(FirstN, n) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()              \
          .WithPerror()

#define ABSL_PLOG_IF_EVERY_POW_2_IMPL(severity, condition)              \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, condition)(EveryPow2) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()              \
          .WithPerror()

#define ABSL_PLOG_IF_EVERY_N_SEC_IMPL(severity, condition, n_seconds)   \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, condition)(EveryNSec, \
                                                             n_seconds) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()              \
          .WithPerror()

#ifndef NDEBUG
#define ABSL_DLOG_IF_EVERY_N_IMPL(severity, condition, n)               \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, condition)(EveryN, n) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_DLOG_IF_FIRST_N_IMPL(severity, condition, n)               \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, condition)(FirstN, n) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_DLOG_IF_EVERY_POW_2_IMPL(severity, condition)              \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, condition)(EveryPow2) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_DLOG_IF_EVERY_N_SEC_IMPL(severity, condition, n_seconds)   \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, condition)(EveryNSec, \
                                                             n_seconds) \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#else  // def NDEBUG
#define ABSL_DLOG_IF_EVERY_N_IMPL(severity, condition, n)                \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, false && (condition))( \
      EveryN, n) ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_DLOG_IF_FIRST_N_IMPL(severity, condition, n)                \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, false && (condition))( \
      FirstN, n) ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_DLOG_IF_EVERY_POW_2_IMPL(severity, condition)               \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, false && (condition))( \
      EveryPow2) ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()

#define ABSL_DLOG_IF_EVERY_N_SEC_IMPL(severity, condition, n_seconds)    \
  ABSL_LOG_INTERNAL_CONDITION##severity(STATEFUL, false && (condition))( \
      EveryNSec, n_seconds)                                              \
      ABSL_LOGGING_INTERNAL_LOG##severity.InternalStream()
#endif  // def NDEBUG

#endif  // ABSL_LOG_INTERNAL_LOG_IMPL_H_
