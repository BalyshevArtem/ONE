/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "SimpleMemoryManager.h"

namespace luci_interpreter
{

uint8_t *SimpleMemoryManager::allocate_memory(const  circle::Tensor &tensor)
{
//  if (!tensor.is_allocatable())
//  {
//    return;
//  }
//  if (tensor.is_data_allocated())
//  {
//    release_memory(tensor);
//  }
  const auto element_size = getDataTypeSize(Tensor::element_type(&tensor));
  const auto num_elements = Tensor::num_elements(&tensor);

  auto *data = new uint8_t[num_elements * element_size];
  return data;
}

void SimpleMemoryManager::release_memory(uint8_t *data)
{
  //assert(data != nullptr);
  if (data == nullptr)
    return;

//  if (!tensor.is_data_allocated())
//  {
//    //tensor.set_data_buffer(nullptr);
//    return;
//  }
  delete[] data;
}

} // namespace luci_interpreter
