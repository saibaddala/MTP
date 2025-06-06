/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#ifndef TENSORFLOW_CORE_SUMMARY_SCHEMA_H_
#define TENSORFLOW_CORE_SUMMARY_SCHEMA_H_

#include "status.h"
#include "sqlite.h"

namespace tensorflow {

constexpr uint32 kTensorboardSqliteApplicationId = 0xfeedabee;

/// \brief Creates TensorBoard SQLite tables and indexes.
///
/// If they are already created, this has no effect. If schema
/// migrations are necessary, they will be performed with logging.
Status SetupTensorboardSqliteDb(Sqlite* db);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_SUMMARY_SCHEMA_H_
