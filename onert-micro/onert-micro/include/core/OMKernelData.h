/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef ONERT_MICRO_CORE_KERNEL_DATA_H
#define ONERT_MICRO_CORE_KERNEL_DATA_H

#include "OMStatus.h"
#include "reader/OMCircleReader.h"

namespace onert_micro
{
namespace core
{

struct DataFullyConnected
{
  int32_t input_offset;
  int32_t weights_offset;
  int32_t output_offset;
  int32_t output_multiplier;
  int output_shift;
  // uint8_t, etc, activation params.
  int32_t quantized_activation_min;
  int32_t quantized_activation_max;
  // Mark the operands as cacheable if they are unchanging, e.g. weights.
  bool lhs_cacheable;
  bool rhs_cacheable;
  // float activation params.
  float float_activation_min;
  float float_activation_max;
};

struct DataAbs
{
// Empty
};

struct Datacustom_gru
{
// Empty
};

} // core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_KERNEL_DATA_H
