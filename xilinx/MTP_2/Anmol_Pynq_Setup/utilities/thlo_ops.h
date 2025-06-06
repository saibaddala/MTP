/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

// This file defines the operations used in the THLO dialect.

#ifndef MLIR_HLO_THLO_IR_THLO_OPS_H
#define MLIR_HLO_THLO_IR_THLO_OPS_H

#include "BuiltinTypes.h"
#include "Dialect.h"
#include "MLIRContext.h"
#include "ControlFlowInterfaces.h"
#include "DestinationStyleOpInterface.h"
#include "InferTypeOpInterface.h"
#include "SideEffectInterfaces.h"
#include "TilingInterface.h"

// Generated dialect declarations.
#include "thlo_dialect.h.inc"

// Generated operation classes.
#define GET_OP_CLASSES
#include "thlo_ops.h.inc"

#endif  // MLIR_HLO_THLO_IR_THLO_OPS_H
