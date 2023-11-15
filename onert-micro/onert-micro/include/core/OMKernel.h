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
  std::vector<uint16_t> _operators;
  OMKernelType _kernel_type = Normal;
  OMBuilderID _builder_id;
  uint8_t *_kernel_data = nullptr;

  void unsetKernelData()
  {
    delete[] _kernel_data;
  }

public:
  OMKernel() = default;
  OMKernel(const OMKernel&) = delete;
  OMKernel(OMKernel &&rhs)
  {
    rhs._operators = std::move(_operators);
    rhs._kernel_data = _kernel_data;
    _kernel_data = nullptr;
    rhs._kernel_type = _kernel_type;
    rhs._builder_id = _builder_id;
  }
  OMKernel &operator=(const OMKernel &) = delete;
  OMKernel &&operator=(const OMKernel &&) = delete;
  ~OMKernel()
  {
    _operators.clear();
    delete[] _kernel_data;
  }

  std::vector<uint16_t> &getKernelOperators()
  {
    return _operators;
  }

  void setKernelOperators(std::vector<uint16_t> &&operators)
  {
    _operators.clear();
    _operators = std::move(operators);
  }

  OMKernelType getKernelType()
  {
    return _kernel_type;
  }

  void setKernelType(OMKernelType kernel_type)
  {
    _kernel_type = kernel_type;
  }

  OMBuilderID getBuilderID()
  {
    return _builder_id;
  }

  void setBuilderID(OMBuilderID builder_id)
  {
    _builder_id = builder_id;
  }

  void setKernelData(uint8_t *kernel_data)
  {
    unsetKernelData();
    _kernel_data = kernel_data;
  }

  uint8_t *getKernelData()
  {
    return _kernel_data;
  }
};

} // core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_KERNEL_H
