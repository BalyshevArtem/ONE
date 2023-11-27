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
#include "core/OMDataType.h"

using namespace onert_micro::core::memory;
using namespace onert_micro;

OMStatus OMRuntimeAllocator::deallocateOutputs(OMRuntimeContext *context, OMRuntimeStorage *storage)
{
  auto &reader = context->getCircleReader();

  const auto outputs = reader.outputs();

  for (uint32_t i = 0; i < outputs->size(); ++i)
  {
    auto tensor_index = outputs->operator[](i);

    uint8_t *allocated_data = nullptr;
    OMStatus status = storage->getDataByTensorIndex(&allocated_data, tensor_index);
    if (status != Ok)
      return status;

    OMMemoryManager::deallocateMemory(allocated_data);
  }

  storage->clearTensorIndexToData();

  return Ok;
}

OMStatus OMRuntimeAllocator::allocate(size_t kernel_index, OMRuntimeContext *context, OMRuntimeStorage *storage)
{
  assert(kernel_index < _alloc_plan.size() && "Wrong kernel index");
  if (kernel_index >= _alloc_plan.size())
    return UnknownError;

  const std::vector<uint16_t> &current_allocate_plan = _alloc_plan[kernel_index];

  for (const uint16_t tensor_index : current_allocate_plan)
  {
    const circle::Tensor *tensor = context->getTensorByIndex(tensor_index);
    const OMShape tensor_shape(tensor);

    int32_t num_elements = tensor_shape.num_elements();
    assert(num_elements >= 0 && "Num elements should be positive");
    if (num_elements < 0)
      return UnknownError;
    const auto casted_num_elements = static_cast<uint32_t>(num_elements);
    const auto type_size = static_cast<uint32_t>(getOMDataTypeSize(onerMicroDatatype(tensor->type())));

    // clear if it is already saved data
    uint8_t *allocated_data = nullptr;
    OMStatus status = storage->getDataByTensorIndex(&allocated_data, tensor_index);
    if (status != Ok)
      return status;

    OMMemoryManager::deallocateMemory(allocated_data);

    // allocate data
    status = OMMemoryManager::allocateMemory(casted_num_elements * type_size, &allocated_data);
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
    uint8_t *allocated_data = nullptr;
    OMStatus status = storage->getDataByTensorIndex(&allocated_data, tensor_index);
    if (status != Ok)
      return status;

    status = OMMemoryManager::deallocateMemory(allocated_data);
    if (status != Ok)
      return status;

    status = storage->removeTensorFromTensorIndexToData(tensor_index);
    if (status != Ok)
      return status;
  }

  return Ok;
}

OMStatus OMRuntimeAllocator::allocateGraphInputs(OMRuntimeContext *context, OMRuntimeStorage *storage)
{
  OMStatus status = Ok;
  auto &reader = context->getCircleReader();
  const auto &graph_inputs = reader.inputs();

  for (uint i = 0; i < graph_inputs->size(); ++i)
  {
    auto tensor_index = graph_inputs->operator[](i);
    const circle::Tensor *tensor = context->getTensorByIndex(tensor_index);
    const OMShape tensor_shape(tensor);

    int32_t num_elements = tensor_shape.num_elements();
    assert(num_elements >= 0 && "Num elements should be positive");
    if (num_elements < 0)
      return UnknownError;
    const auto casted_num_elements = static_cast<uint32_t>(num_elements);
    const auto type_size = static_cast<uint32_t>(getOMDataTypeSize(onerMicroDatatype(tensor->type())));

    uint8_t *allocated_data = nullptr;
    // First clear if already allocated
    status = storage->getDataByTensorIndex(&allocated_data, tensor_index);
    if (status != Ok)
      return status;

    OMMemoryManager::deallocateMemory(allocated_data);

    // Then Allocate
    status = OMMemoryManager::allocateMemory(casted_num_elements * type_size, &allocated_data);
    if (status != Ok)
      return status;

    storage->saveDataToTensorIndex(allocated_data, tensor_index);
  }

  return status;
}

