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
#include "mbed.h"
#undef ARG_MAX
#define LUCI_LOG 0
#include <luci_interpreter/Interpreter.h>
#include <luci_interpreter/GraphBuilderRegistry.h>
#include <luci_interpreter/StaticMemoryManager.h>
//#include <luci/Importer.h>
//#include <luci/IR/Module.h>
#include <loco/IR/DataTypeTraits.h>
#include "rev2.h"
#include <cstdlib>
#include <iostream>
#include <luci/Log.h>
#include <stm32h7xx_hal.h>
#include "mbed_mem_trace.h"

void fill_in_tensor(std::vector<char> &data, loco::DataType dtype)
{
  switch (dtype)
  {
    case loco::DataType::FLOAT32:
      std::cout << "loco::DataType::FLOAT32\n";
      for (int i = 0; i < data.size() / sizeof(float); ++i)
      {
        reinterpret_cast<float *>(data.data())[i] = 123.f;
      }
      break;
    case loco::DataType::S8:
      std::cout << "loco::DataType::S8\n";
      for (int i = 0; i < data.size() / sizeof(int8_t); ++i)
      {
        reinterpret_cast<int8_t *>(data.data())[i] = 123;
      }
      break;
    case loco::DataType::U8:
      std::cout << "loco::DataType::U8\n";
      for (int i = 0; i < data.size() / sizeof(uint8_t); ++i)
      {
        reinterpret_cast<uint8_t *>(data.data())[i] = 123;
      }
      break;
    default:
      assert(false && "Unsupported type");
  }
}
mbed_stats_heap_t heap_info;
void print_memory_stats()
{
  printf("\nMemoryStats:");
  mbed_stats_heap_get(&heap_info);
  printf("\n\tBytes allocated currently: %ld", heap_info.current_size);
  printf("\n\tMax bytes allocated at a given time: %ld", heap_info.max_size);
  printf("\n\tCumulative sum of bytes ever allocated: %ld", heap_info.total_size);
  printf("\n\tCurrent number of bytes allocated for the heap: %ld", heap_info.reserved_size);
  printf("\n\tCurrent number of allocations: %ld", heap_info.alloc_cnt);
  printf("\n\tNumber of failed allocations: %ld", heap_info.alloc_fail_cnt);
}

int main()
{
  print_memory_stats();
  std::cout << "\nSystemCoreClock " << SystemCoreClock << "\n";
  flatbuffers::Verifier verifier{reinterpret_cast<const uint8_t *>(circle_model_raw), sizeof(circle_model_raw) / sizeof(circle_model_raw[0])};

  std::cout << "circle::VerifyModelBuffer\n";
  if (!circle::VerifyModelBuffer(verifier))
  {
    std::cout << "ERROR: Failed to verify circle\n";
  }
  std::cout << "OK\n";
  std::cout << "circle::GetModel(circle_model_raw)\n";
  //auto model = circle::GetModel(circle_model_raw);
  std::cout << "luci::Importer().importModule\n";
  //const auto optimized_source = luci_interpreter::source_without_constant_copying();
  //auto module = luci::Importer(optimized_source.get()).importModule(model);
  //auto module = luci::Importer().importModule(model);
  luci_interpreter::Interpreter interpreter(const_cast<char *>(circle_model_raw));
 // print_memory_stats();
  std::cout << "OK\n";
  std::cout << "std::make_unique<luci_interpreter::Interpreter>(module.get())\n";

  //auto interpreter = std::make_unique<luci_interpreter::Interpreter>(module.get());

  Timer t;
 // print_memory_stats();
  std::cout << "OK\n";
 // auto nodes = module->graph()->nodes();
 // auto nodes_count = nodes->size();

  const auto input_tensors = interpreter.getInputTensors();
  for (int i = 0; i < input_tensors.size(); ++i)
  {
    //auto *node = dynamic_cast<luci::CircleNode *>(nodes->at(i));
   // assert(node);
   // if (node->opcode() == luci::CircleOpcode::CIRCLEINPUT)
   // {
    //  auto *input_node = static_cast<luci::CircleInput *>(node);
    //  loco::GraphInput *g_input = module->graph()->inputs()->at(input_node->index());
    const auto input_tensor = input_tensors.at(i);
    const luci_interpreter::Shape shape = input_tensor->shape();//g_input->shape();
    size_t data_size = 1;
    for (int d = 0; d < shape.num_dims(); ++d)
    {
     // assert(shape.dim(d).known());
      data_size *= shape.dim(d);
    }
    data_size *= loco::size(input_tensor->element_type());

    std::vector<char> data(data_size);
    fill_in_tensor(data, input_tensor->element_type());
    interpreter.write_input_tensor(input_tensor, data.data(), data_size);
      //interpreter->writeInputTensor(static_cast<luci::CircleInput *>(node), data.data(),
      //                              data_size);
  }

    print_memory_stats();
    //t.start();
    interpreter.interpret();
    //t.stop();
    print_memory_stats();
    std::cout << "\nFinished in " << t.read_us() << "\n";
}
