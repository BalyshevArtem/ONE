/*
 * Copyright (c) 2020 Samsung Electronics Co., Ltd. All Rights Reserved
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

#ifndef LUCI_INTERPRETER_CORE_RUNTIMEMODULE_H
#define LUCI_INTERPRETER_CORE_RUNTIMEMODULE_H

#include "core/RuntimeGraph.h"
#include "memory_managers/MemoryManager.h"
#include "luci_interpreter/core/reader/CircleMicroReader.h"

#include <memory>
#include <vector>

namespace luci_interpreter
{

class RuntimeModule
{
public:
  RuntimeModule() = default;

  RuntimeGraph *addGraph(IMemoryManager *memory_manager, bool use_static_allocations)
  {
//    if (use_static_allocations)
//      _graphs.push_back(std::make_unique<StaticRuntimeGraph>(memory_manager, &_circle_reader));
//    else
//      _graphs.emplace_back(memory_manager, &_circle_reader);
//    return &_graphs.back();
    _graphs = std::make_unique<RuntimeGraph>(memory_manager, &_circle_reader);
  }

//  RuntimeGraph *getRuntimeGraphAt(uint32_t pos)
//  {
//    return &_graphs.at(pos);
//  }

  std::vector<Tensor *> getInputTensors() const { return _graphs->getInputTensors(); }
  std::vector<Tensor *> getOutputTensors() const
  {
    return _graphs->getOutputTensors();
  }

  void execute() const { getMainGraph()->execute(); }

  CircleReader &getCircleReader()
  {
    return _circle_reader;
  }

  IBaseRuntimeGraph *getMainGraph() const { return _graphs.get(); }

private:

  std::unique_ptr<RuntimeGraph> _graphs;

  CircleReader _circle_reader;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_CORE_RUNTIMEMODULE_H
