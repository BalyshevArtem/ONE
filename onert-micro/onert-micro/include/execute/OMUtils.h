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

#ifndef ONERT_MICRO_EXECUTE_UTILS_H
#define ONERT_MICRO_EXECUTE_UTILS_H

#include "OMStatus.h"
#include "core/reader/OMCircleReader.h"
#include "core/OMRuntimeShape.h"

namespace onert_micro
{
namespace execute
{

template <typename T>
OMStatus calculateActivationRange(circle::ActivationFunctionType activation, T *activation_min, T *activation_max)
{
  switch (activation)
  {
    case circle::ActivationFunctionType::ActivationFunctionType_NONE:
      *activation_min = std::numeric_limits<T>::lowest();
      *activation_max = std::numeric_limits<T>::max();
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU:
      *activation_min = 0;
      *activation_max = std::numeric_limits<T>::max();
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU_N1_TO_1:
      *activation_min = -1;
      *activation_max = 1;
      break;
    case circle::ActivationFunctionType::ActivationFunctionType_RELU6:
      *activation_min = 0;
      *activation_max = 6;
      break;
    default:
      assert(false && "Unsupported activation.");
      return UnsupportedActivation;
  }

  return Ok;
}

inline double getQuantizedConvolutionMultipler(float input_scale, float filter_scale,
                                               float output_scale)
{
  const double input_product_scale = static_cast<double>(input_scale * filter_scale);

  assert(input_product_scale >= 0);

  return input_product_scale / static_cast<double>(output_scale);
}

void quantizeMultiplier(double double_multiplier, int32_t *quantized_multiplier, int *shift);

void calculateActivationRangeQuantized(circle::ActivationFunctionType activation, int32_t output_zero_point,
                                       float output_scale, circle::TensorType data_type,
                                       int32_t *activation_min, int32_t *activation_max);

} // execute
} // namespace onert_micro

#endif // ONERT_MICRO_EXECUTE_UTILS_H

