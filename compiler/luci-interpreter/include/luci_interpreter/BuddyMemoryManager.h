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

#include "luci_interpreter/MemoryManager.h"

#ifndef LUCI_INTERPRETER_BUDDY_MEMORY_MANAGER_H
#define LUCI_INTERPRETER_BUDDY_MEMORY_MANAGER_H

namespace luci_interpreter
{

class BuddyMemoryManager : public MManager
{
public:
  BuddyMemoryManager(uint8_t *memory_start, int memSize);

  void allocate_memory(luci_interpreter::Tensor *tensor) final;
  void release_memory(luci_interpreter::Tensor *tensor) final;

private:
  struct TBlock {
    TBlock * nextFree;
    TBlock * self;
    bool free;
    int size;
  };
  TBlock * g_Start;
  int g_Blocks;
  int g_Size;
  TBlock * g_Free[32];

  int powof2(int val)
  {
    int i = 0;

    while(val >>= 1)
      i++;

    return i;
  }

  void addToList(TBlock * block, int l)
  {
    if(!block)
      return;

    block->nextFree = g_Free[l];
    g_Free[l] = block;
  }

  void removeFromList(TBlock * block, int l)
  {
    if(!block)
      return;

    TBlock * tmp = g_Free[l];

    if(block == tmp){
      g_Free[l] = block->nextFree;
      return;
    }

    while(tmp){
      if(tmp->nextFree == block)
        tmp->nextFree = block->nextFree;

      tmp = tmp->nextFree;
    }
  }

  void HeapInit(void * memPool, int memSize)
  {
    int p = powof2(memSize);
    memSize = 1 << p;

    g_Start = (TBlock *) memPool;
    g_Start->size = memSize - sizeof(TBlock);
    g_Start->free = true;
    g_Start->self = g_Start;

    g_Blocks = 0;
    g_Size = g_Start->size;

    for(int i = 0; i < 32; i++)
      g_Free[i] = NULL;

    addToList(g_Start, p);
  }

  TBlock * divide(TBlock * block, int l)
  {
    int size = ((block->size + sizeof(TBlock)) / 2) - sizeof(TBlock);

    removeFromList(block, l);

    block->free = true;
    block->size = size;
    block->self = block;

    TBlock * buddy;
    buddy = (TBlock *) ((uint8_t *)block + sizeof(TBlock) + size);
    buddy->free = true;
    buddy->size = size;
    buddy->self = buddy;

    addToList(buddy, l - 1);

    return block;
  }

  TBlock * findBuddy(TBlock * block, int l)
  {
    long addr = ((uint8_t *) block - (uint8_t *) g_Start);

    return (TBlock *)((addr ^= (1 << l)) + (size_t) g_Start);
  }

  TBlock * merge(TBlock * block)
  {
    TBlock * buddy;

    int l = powof2(block->size + sizeof(TBlock));

    buddy = findBuddy(block, l);

    if(!buddy->free || buddy->size != block->size)
      return NULL;

    if(block > buddy){
      TBlock * x = block;
      block = buddy;
      buddy = x;
    }

    removeFromList(block, l);
    removeFromList(buddy, l);

    block->size = block->size * 2 + sizeof(TBlock);
    block->free = true;
    block->self = block;

    addToList(block, l + 1);

    return block;
  }
};

}// namespace luci_interpreter

#endif // LUCI_INTERPRETER_BUDDY_MEMORY_MANAGER_H