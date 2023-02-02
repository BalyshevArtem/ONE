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

#include "ModuleLoader.h"

#include "GraphLoader.h"

namespace luci_interpreter
{
//ModuleLoader::ModuleLoader(RuntimeModule *runtime_module,
//                           IMemoryManager *memory_manager)
//  : _runtime_module(runtime_module),
//    _memory_manager(memory_manager)//, _index_to_tensor(std::unordered_map<const circle::Tensor *, Tensor *>{})
//{
//}

void ModuleLoader::load(RuntimeModule *runtime_module,
                        IMemoryManager *memory_manager,
                        bool use_static_memory_manager,
                        const char *model_data_raw)
{
  const circle::Model *model = circle::GetModel(model_data_raw);

  CircleReader &reader = runtime_module->getCircleReader();
  if (!reader.parse(model))
    assert(false && "Error during parse");

  assert(reader.num_subgraph() == 1);

//  for (size_t i = 0; i < reader.num_subgraph(); ++i)
//  {
  runtime_module->addGraph(memory_manager, use_static_memory_manager);
//  }

  //for (size_t i = 0; i < reader.num_subgraph(); ++i)
  //{
    if (!reader.select_subgraph(0))
      assert(false && "Error during select subgraph");
    IBaseRuntimeGraph *runtime_graph = runtime_module->getMainGraph(); //_runtime_graphs.at(i);
   // GraphLoader loader(&reader, runtime_graph, memory_manager);//, &_index_to_tensor);

    GraphLoader::loadTensors(&reader, runtime_graph, use_static_memory_manager);
   // loader.initInputOutputTensors(use_static_memory_manager);
    GraphLoader::loadOperators(&reader, runtime_graph, use_static_memory_manager);
  //}

  // For Dynamic Memory manager we build memory allocate/deallocate plan and then configure kernels.
  // For Static Memory manager we only configure kernels.
  if (not use_static_memory_manager)
  {
    // Dynamic memory manager case
    //RuntimeGraph *runtime_graph = runtime_module->getMainGraph();

   // for (size_t i = 0; i < reader.num_subgraph(); ++i)
    //{
    //  runtime_graph = runtime_module->getRuntimeGraphAt(i);
      runtime_graph->configure();

     // if (memory_manager->is_allocate_input())
        runtime_graph->configureGraphInputs();
   // }
  }
  else
  {
    assert(false);
//    // Static memory manager case
//    for (size_t i = 0; i < reader.num_subgraph(); ++i)
//    {
//      _runtime_graphs.at(i)->configure_kernels();
//      // TODO add mem allocate inpout!
//    }
  }
}

} // namespace luci_interpreter
