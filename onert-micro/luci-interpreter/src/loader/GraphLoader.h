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

#ifndef LUCI_INTERPRETER_LOADER_GRAPHLOADER_H
#define LUCI_INTERPRETER_LOADER_GRAPHLOADER_H

#include "core/RuntimeGraph.h"
#include "memory_managers/MemoryManager.h"
#include "luci_interpreter/core/reader/CircleMicroReader.h"

#include <unordered_map>

namespace luci_interpreter
{

class GraphLoader
{
public:
  GraphLoader() = default;//(CircleReader *reader, IBaseRuntimeGraph *runtime_graph,
             // IMemoryManager *memory_manager);
             // std::unordered_map<const circle::Tensor *, Tensor *> *index_to_tensor);

  static void loadTensors(CircleReader *reader, IBaseRuntimeGraph *runtime_graph, bool use_static_memory_manager);
  //static void initInputOutputTensors(bool use_static_memory_manager) const;
  static void loadOperators(CircleReader *reader, IBaseRuntimeGraph *runtime_graph, bool use_static_memory_manager);

private:
  static bool isCouldBeEmplaceTensor(CircleReader *reader, const int32_t tensor_index);

 // IBaseRuntimeGraph *_runtime_graph;
 // IMemoryManager *_memory_manager;
 // CircleReader *_reader;
  //std::unordered_map<const circle::Tensor *, Tensor *> *_index_to_tensor;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_LOADER_GRAPHLOADER_H
