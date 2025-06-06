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

#ifndef MLIR_HLO_DEALLOACTION_DEALLOCATION_OPS_H
#define MLIR_HLO_DEALLOACTION_DEALLOCATION_OPS_H

#include "Attributes.h"
#include "BuiltinTypes.h"
#include "Dialect.h"
#include "MLIRContext.h"
#include "OpDefinition.h"
#include "OpImplementation.h"
#include "SideEffectInterfaces.h"

#define GET_TYPEDEF_CLASSES
#include "deallocation_typedefs.h.inc"
#undef GET_TYPEDEF_CLASSES

#define GET_OP_CLASSES
#include "deallocation_dialect.h.inc"
#include "deallocation_ops.h.inc"
#undef GET_OP_CLASSES

#endif  // MLIR_HLO_DEALLOACTION_DEALLOCATION_OPS_H
