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

#ifndef ONERT_MICRO_CORE_SISO_RUNTIME_KERNEL_H
#define ONERT_MICRO_CORE_SISO_RUNTIME_KERNEL_H

#include "core/reader/OMCircleReader.h"
#include "core/OMKernel.h"
#include "core/OMRuntimeContext.h"
#include "core/OMRuntimeStorage.h"

#include <cstdint>

namespace onert_micro
{
namespace core
{

class OMSISORuntimeKernel
{
public:
  OMSISORuntimeKernel() = delete;
  OMSISORuntimeKernel(core::OMKernel &kernel, core::OMRuntimeContext &runtime_context);
  OMSISORuntimeKernel(const OMSISORuntimeKernel &) = delete;
  OMSISORuntimeKernel(OMSISORuntimeKernel&&) = delete;
  ~OMSISORuntimeKernel() = default;
  OMSISORuntimeKernel &operator=(const OMSISORuntimeKernel &) = delete;
  OMSISORuntimeKernel &&operator=(const OMSISORuntimeKernel &&) = delete;

public:
  OMStatus getDataFromStorage(core::OMKernel &kernel,
                              core::OMRuntimeStorage &storage,
                              uint8_t **input_data,
                              uint8_t **output_data);

public:
  const circle::Tensor *input;
  const circle::Tensor *output;
  int32_t input_index;
  int32_t output_index;
};

} // core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_SISO_RUNTIME_KERNEL_H
