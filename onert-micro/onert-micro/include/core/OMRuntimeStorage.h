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

#ifndef ONERT_MICRO_CORE_RUNTIME_STORAGE_H
#define ONERT_MICRO_CORE_RUNTIME_STORAGE_H

#include "OMRuntimeShape.h"
#include "OMKernel.h"
#include "OMStatus.h"

#include <vector>
#include <unordered_map>
#include <cstdint>

namespace onert_micro
{
namespace core
{

class OMRuntimeStorage
{
private:
  std::unordered_map<uint16_t, OMRuntimeShape> _tensor_index_to_runtime_shape;
  std::vector<OMKernel> _kernels;
  std::unordered_map<uint16_t, uint8_t *> _tensor_index_to_data;

public:
  OMRuntimeStorage() = default;
  OMRuntimeStorage(const OMRuntimeStorage &) = delete;
  OMRuntimeStorage(OMRuntimeStorage &&) = default;
  OMRuntimeStorage &operator=(const OMRuntimeStorage &) = delete;
  OMRuntimeStorage &&operator=(const OMRuntimeStorage &&) = delete;
  // TODO check it
  ~OMRuntimeStorage() = default;

  OMStatus saveDataToTensorIndex(void *data, uint16_t tensor_index);
  OMStatus getDataByTensorIndex(void *data, uint16_t tensor_index);

  OMStatus saveKernels(std::vector<OMKernel> &&kernels)
  {
    _kernels.clear();
    _kernels = std::move(kernels);
    return Ok;
  }

  std::vector<OMKernel> &getKernels()
  {
    return _kernels;
  }

  //OMStatus saveRuntimeShapeToTensorIndex(void *data, uint16_t tensor_index);
  //OMStatus getRuntimeShapeByTensorIndex(void *data, uint16_t tensor_index);
};

} // core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_RUNTIME_STORAGE_H

