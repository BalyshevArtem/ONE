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

#ifndef ONERT_MICRO_CORE_RUNTIME_GRAPH_H
#define ONERT_MICRO_CORE_RUNTIME_GRAPH_H

#include "OMStatus.h"

#include "OMRuntimeModule.h"
#include "OMRuntimeContext.h"
#include "OMRuntimeStorage.h"
#include "memory/OMRuntimeAllocator.h"

#include <cstdint>

namespace onert_micro
{
namespace core
{

class OMRuntimeGraph
{
private:
  OMRuntimeContext _context;
  OMRuntimeStorage _storage{};
  memory::OMRuntimeAllocator _allocator{};

public:
  explicit OMRuntimeGraph(uint32_t graph_index): _context(OMRuntimeContext(graph_index))
  {
    // Do nothing
  }
  OMRuntimeGraph() = delete;
  OMRuntimeGraph(const OMRuntimeGraph &) = delete;
  OMRuntimeGraph(OMRuntimeGraph &&) = delete;
  ~OMRuntimeGraph() = default;

  int getNumberOfInputs();
  int getNumberOfOutput();

  uint8_t *getInputDataAt(int position);
  uint8_t *getOutputDataAt(int position);

  OMStatus run();
  OMStatus configure();
};

} // core
} // namespace onert_micro

#endif // ONERT_MICRO_CORE_RUNTIME_GRAPH_H
