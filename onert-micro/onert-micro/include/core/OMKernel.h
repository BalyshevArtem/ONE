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

#ifndef ONERT_MICRO_CORE_KERNEL_H
#define ONERT_MICRO_CORE_KERNEL_H

#include "OMKernelType.h"

#include <vector>
#include <cstdint>

namespace onert_micro
{
namespace core
{

class OMKernel
{
private:
  std::vector<uint8_t> _operators{};
  std::vector<uint16_t> _inputs_indexes{};
  std::vector<uint16_t> _outputs_indexes{};
  // TODO: pointer to configure function
  // TODO: pointer to execute function
  OMKernelType _kernel_type = Normal;
  bool _is_inplace = false;
  void *_data = nullptr;

public:
  OMKernel() = default;
  OMKernel(const OMKernel&) = delete;
  OMKernel(OMKernel &&) = delete;
  OMKernel &operator=(const OMKernel &) = delete;
  OMKernel &&operator=(const OMKernel &&) = delete;
  ~OMKernel() = default;

  const std::vector<uint8_t> &getKernelOperators()
  {
    return _operators;
  }

  const std::vector<uint16_t> &getKernelInputs()
  {
    return _inputs_indexes;
  }

  const std::vector<uint16_t> &getKernelOutputs()
  {
    return _outputs_indexes;
  }
};

} // core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_KERNEL_H
