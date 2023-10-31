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

#include "core/memory/OMRuntimeAllocator.h"
#include "core/memory/OMMemoryManager.h"
#include "core/OMShape.h"

using namespace onert_micro::core::memory;
using namespace onert_micro;

OMStatus OMRuntimeAllocator::allocate(size_t kernel_index, OMRuntimeContext *context, OMRuntimeStorage *storage)
{
  assert(kernel_index < _alloc_plan.size() && "Wrong kernel index");
  if (kernel_index >= _alloc_plan.size())
    return UnknownError;

  const std::vector<uint16_t> &current_allocate_plan = _alloc_plan[kernel_index];

  for (const uint16_t tensor_index : current_allocate_plan)
  {
    const circle::Tensor *tensor = context->getCircleTensorByIndex(tensor_index);
    const OMShape tensor_shape(tensor);

    int32_t num_elements = tensor_shape.num_elements();
    assert(num_elements >= 0 && "Num elements should be positive");
    if (num_elements < 0)
      return UnknownError;
    const uint32_t casted_num_elements = static_cast<uint32_t>(num_elements);
    const uint32_t type_size = static_cast<uint32_t>(sizeof(tensor->type()));

    void *allocated_data = nullptr;
    OMStatus status = OMMemoryManager::allocateMemory(casted_num_elements * type_size, allocated_data);
    if (status != Ok)
      return status;

    storage->saveDataToTensorIndex(allocated_data, tensor_index);
  }

  return Ok;
}

OMStatus OMRuntimeAllocator::deallocate(size_t kernel_index, OMRuntimeStorage *storage)
{
  assert(kernel_index < _alloc_plan.size() && "Wrong kernel index");
  if (kernel_index >= _alloc_plan.size())
    return UnknownError;

  const std::vector<uint16_t> &current_deallocate_plan = _dealloc_plan[kernel_index];

  for (const uint16_t tensor_index : current_deallocate_plan)
  {
    void *allocated_data = nullptr;
    OMStatus status = storage->getDataByTensorIndex(allocated_data, tensor_index);
    if (status != Ok)
      return status;

    status = OMMemoryManager::deallocateMemory(allocated_data);
    if (status != Ok)
      return status;
  }

  return Ok;
}

