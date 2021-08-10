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

#include "luci_interpreter/BuddyMemoryManager.h"

namespace luci_interpreter
{

BuddyMemoryManager::BuddyMemoryManager(uint8_t *memory_start, int memSize)
{
  int p = powof2(memSize);
  memSize = 1 << p;

  g_Start = (TBlock *) memory_start;
  g_Start->size = memSize - sizeof(TBlock);
  g_Start->free = true;
  g_Start->self = g_Start;

  g_Blocks = 0;
  g_Size = g_Start->size;

  for(int i = 0; i < 32; i++)
    g_Free[i] = NULL;

  addToList(g_Start, p);
}

void BuddyMemoryManager::allocate_memory(luci_interpreter::Tensor *tensor)
{
  tensor->set_data_buffer(nullptr);

  const size_t element_size = getDataTypeSize(tensor->element_type());
  const int32_t num_elements = tensor->shape().num_elements();

  auto size = num_elements * element_size;

  int l = powof2(size + sizeof(TBlock)) + 1;

  while(!g_Free[l] && l < 32)
    l++;

  assert(l<32);

  TBlock * tmp;
  tmp = g_Free[l];

  removeFromList(tmp, l);

  while((tmp->size + sizeof(TBlock)) / 2 >= size + sizeof(TBlock))
  {
    tmp = divide(tmp, l);
    l--;
  }

  tmp->free = false;
  tmp->self = tmp;
  g_Blocks++;

  uint8_t *data = (uint8_t*)(tmp + 1);
  tensor->set_data_buffer(data);
}

void BuddyMemoryManager::release_memory(luci_interpreter::Tensor *tensor)
{
  auto blk = tensor->data<void>();

  TBlock * tmp = (TBlock *)((uint8_t *) blk - sizeof(TBlock));

  assert(tmp->self == tmp);

  tmp->free = true;

  addToList(tmp, powof2(tmp->size + sizeof(TBlock)));
  while(tmp)
    if(tmp->size == g_Size)
      break;
    else
      tmp = merge(tmp);

  g_Blocks--;

  tensor->set_data_buffer(nullptr);
}

} // namespace luci_interpreter