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

#include "core/RuntimeGraph.h"
//#include "memory_managers/StaticMemoryManager.h"
#include "kernels/KernelBuilder.h"

#include <algorithm>
#include <map>

namespace luci_interpreter
{

// IBaseRuntimeGraph
IBaseRuntimeGraph::IBaseRuntimeGraph(IMemoryManager *memory_manager, CircleReader *circle_reader)
  : _memory_manager(memory_manager),
    _index_to_tensor(std::unordered_map<const circle::Tensor *, std::unique_ptr<Tensor>>{}),
    _reader(circle_reader), _inplace_op_indexes(std::unordered_set<uint32_t>{})
{
}

std::vector<Tensor *> IBaseRuntimeGraph::getOutputTensors() const
{
  std::vector<Tensor *> output_tensors;

  for (const auto output_ind : _reader->outputs())
  {
    const auto raw_tensor = _reader->tensors()[output_ind];
    if (_index_to_tensor.find(raw_tensor) == _index_to_tensor.end())
    {
      assert(false && "Failed import graph input tensor");
    }

    auto tensor_interpreter = _index_to_tensor.at(raw_tensor).get();

    output_tensors.push_back(tensor_interpreter);
  }
  assert(!output_tensors.empty());

  return output_tensors;
}

std::vector<Tensor *> IBaseRuntimeGraph::getInputTensors() const
{
  std::vector<Tensor *> input_tensors;

  for (const auto input_ind : _reader->inputs())
  {
    const auto raw_tensor = _reader->tensors()[input_ind];
    if (_index_to_tensor.find(raw_tensor) == _index_to_tensor.end())
    {
      assert(false && "Failed import graph input tensor");
    }

    auto tensor_interpreter = _index_to_tensor.at(raw_tensor).get();

    input_tensors.push_back(tensor_interpreter);
  }
  assert(!input_tensors.empty());

  return input_tensors;
}

Tensor *IBaseRuntimeGraph::addTensor(const circle::Tensor *raw_tensor, std::unique_ptr<Tensor> &&tensor)
{
  assert(raw_tensor != nullptr);
  assert(tensor != nullptr);
  //_tensors.push_back(std::move(tensor));
  //return _tensors.back().get();
  _index_to_tensor[raw_tensor] = std::move(tensor);
  return _index_to_tensor[raw_tensor].get();
}

#ifndef DIS_QUANT
AffineQuantization *
IBaseRuntimeGraph::addAffineQuantization(std::unique_ptr<AffineQuantization> &&quantization)
{
  assert(quantization != nullptr);
  _affine_quantizations.push_back(std::move(quantization));
  return _affine_quantizations.back().get();
}

void IBaseRuntimeGraph::addIntermediateTensorAffineQuantization(
  AffineQuantization *intermediate_tensor_affine_quant)
{
  _intermediate_tensors_affine_quantizations.push_back(intermediate_tensor_affine_quant);
}

#endif

//void IBaseRuntimeGraph::addInputTensor(Tensor *input_tensor, int pos)
//{
//  _input_tensors.at(pos) = input_tensor;
//}
//
//
//void IBaseRuntimeGraph::addOutputTensor(Tensor *output_tensor, int pos)
//{
//  _output_tensors.at(pos) = output_tensor;
//}

void IBaseRuntimeGraph::configureAllocations(Tensor *tensor)
{
  _memory_manager->allocate_memory(*tensor);
}

//void IBaseRuntimeGraph::addKernel(std::unique_ptr<Kernel> &&kernel)
//{
//  assert(kernel != nullptr);
//  _kernels.push_back(std::move(kernel));
//  _is_valid = false;
//}

// RuntimeGraph
RuntimeGraph::RuntimeGraph(IMemoryManager *memory_manager,
                           CircleReader *circle_reader) : IBaseRuntimeGraph(memory_manager,
                                                                            circle_reader) {}

RuntimeGraph::~RuntimeGraph()
{
  for (auto &idx_to_tensor : _index_to_tensor)
  {
    const auto tensor = idx_to_tensor.second.get();
    if (tensor->is_data_allocated())
      _memory_manager->release_memory(*tensor);
  }
}

void RuntimeGraph::buildAllocDeallocPlan()
{
  invalidate();
  using Lifetime = std::pair<size_t, size_t>;
  std::map<Tensor *, Lifetime> lifetimes;
  const size_t num_kernels = _reader->operators().size(); //_kernels.size();
  for (int32_t index = 0; index < num_kernels; ++index)
  {
    const auto kernel = _reader->operators().at(index);
    assert(kernel != nullptr);
    // const auto &kernel = _kernels[index];
    for (int32_t j = 0; j < kernel->inputs()->size(); ++j)
    {
      const auto input_index = kernel->inputs()->operator[](j);

      if (input_index == -1)
        continue;

      const auto raw_tensor = _reader->tensors()[input_index];

      //assert(_index_to_tensor.find(raw_tensor) != _index_to_tensor.end());
      if (_index_to_tensor.find(raw_tensor) == _index_to_tensor.end())
        continue;

      auto tensor = _index_to_tensor[raw_tensor].get();

//      auto i_1 = tensor->dim(0);
//      auto i_2 = tensor->dim(1);

      if (lifetimes.count(tensor) > 0)
      {
        if (_inplace_op_indexes.find(index) != _inplace_op_indexes.end())
          lifetimes.at(tensor).second = -1;
        else
          lifetimes.at(tensor).second = index;
      }
    }

    for (int32_t j = 0; j < kernel->outputs()->size(); ++j)
    {
      const auto output_index = kernel->outputs()->operator[](j);
      const auto raw_tensor = _reader->tensors()[output_index];
      auto tensor = _index_to_tensor[raw_tensor].get();

//      auto i_1 = tensor->dim(0);
//      auto i_2 = tensor->dim(1);

      assert(lifetimes.count(tensor) == 0);
      if (_inplace_op_indexes.find(index) != _inplace_op_indexes.end())
        lifetimes[tensor] = Lifetime(-1, index);
      else
        lifetimes[tensor] = Lifetime(index, index);
    }
  }

  for (const auto output_ind : _reader->outputs())
  {
    const auto raw_tensor = _reader->tensors()[output_ind];
    if (_index_to_tensor.find(raw_tensor) == _index_to_tensor.end())
    {
      assert(false && "Failed import graph input tensor");
    }

    auto tensor = _index_to_tensor.at(raw_tensor).get();

    if (lifetimes.count(tensor) > 0)
      lifetimes.at(tensor).second = num_kernels;
  }
  //
  _alloc_plan.assign(num_kernels, std::vector<Tensor *>());
  _dealloc_plan.assign(num_kernels + 1, std::vector<Tensor *>());
  for (const auto &item : lifetimes)
  {
    if (item.second.first != -1)
      _alloc_plan[item.second.first].push_back(item.first);
    if (item.second.second != -1)
      _dealloc_plan[item.second.second].push_back(item.first);
  }
  _is_valid = true;
}

void RuntimeGraph::allocate(size_t kernel_index) const
{
  assert(_is_valid && kernel_index < _alloc_plan.size());
  for (Tensor *tensor : _alloc_plan[kernel_index])
  {
    _memory_manager->allocate_memory(*tensor);
  }
}

void RuntimeGraph::deallocate(size_t kernel_index) const
{
  assert(_is_valid && kernel_index < _dealloc_plan.size());
  for (Tensor *tensor : _dealloc_plan[kernel_index])
  {
    _memory_manager->release_memory(*tensor);
  }
}

void RuntimeGraph::configureGraphInputs()
{
  for (const auto input_ind : _reader->inputs())
  {
    const auto raw_tensor = _reader->tensors()[input_ind];
    if (_index_to_tensor.find(raw_tensor) == _index_to_tensor.end())
    {
      assert(false && "Failed import graph input tensor");
    }

    auto *tensor_interpreter = _index_to_tensor.at(raw_tensor).get();

    _memory_manager->allocate_memory(*tensor_interpreter);
  }
}

Tensor *IBaseRuntimeGraph::getTensorByIndex(int32_t index)
{
  if (index < 0)
    return nullptr;

  const auto raw_tensor = _reader->tensors()[index];

  if (_index_to_tensor.find(raw_tensor) == _index_to_tensor.end())
  {
    assert(false && "Failed import operation input tensor");
    return nullptr;
  }

  return _index_to_tensor.at(raw_tensor).get();
}

uint8_t *IBaseRuntimeGraph::getDataBufferFromCircleTensor(const circle::Tensor *raw_tensor)
{
  if (raw_tensor == nullptr)
    return nullptr;

  auto const &buffer = wrap(_reader->buffers()[raw_tensor->buffer()]->data());
  return const_cast<uint8_t *>(buffer.data());
}

const circle::Tensor *IBaseRuntimeGraph::getCircleTensorByIndex(int32_t index)
{
  if (index < 0)
    return nullptr;

  const auto raw_tensor = _reader->tensors()[index];

  return raw_tensor;
}

void RuntimeGraph::configure()
{
  KernelConfigureRegistry kernel_configure;

  for (uint32_t i = 0; i < _reader->operators().size(); ++i)
  {
    const auto op = _reader->operators().at(i);
    assert(op != nullptr);

  //  std::vector<const Tensor *> input_tensors(op->inputs()->size());
  //  std::vector<Tensor *> output_tensors(op->outputs()->size());

//    for (int32_t j = 0; j < op->inputs()->size(); ++j)
//    {
//      const auto input_index = op->inputs()->operator[](j);
//      if (input_index != -1)
//      {
//        const auto raw_tensor = _reader->tensors()[input_index];
//
//        if (_index_to_tensor.find(raw_tensor) == _index_to_tensor.end())
//        {
//          assert(false && "Failed import operation input tensor");
//          return;
//        }
//        auto input_tensor = _index_to_tensor.at(raw_tensor).get();
//
//        input_tensors.at(j) = input_tensor;
//      }
//      else
//      {
//        input_tensors.at(j) = nullptr;
//      }
//    }

//    for (int32_t j = 0; j < op->outputs()->size(); ++j)
//    {
//      const auto output_index = op->outputs()->operator[](j);
//      const auto raw_tensor = _reader->tensors()[output_index];
//
//      if (_index_to_tensor.find(raw_tensor) == _index_to_tensor.end())
//      {
//        assert(false && "Failed import operation output tensor");
//        return;
//      }
//      auto output_tensor = _index_to_tensor.at(raw_tensor).get();
//      output_tensors.at(j) = output_tensor;
//    }

    const auto opcode = _reader->builtin_code(op);

    kernel_configure.configure_kernel(op, opcode, this);
  }

  if (not _is_valid)
    buildAllocDeallocPlan();

  _is_valid = true;
}

void RuntimeGraph::execute()
{
  if (not _is_valid)
    configure();

  KernelExecuteRegistry kernel_executor;

  for (uint32_t i = 0; i < _reader->operators().size(); ++i)
  {
    const auto op = _reader->operators().at(i);
    assert(op != nullptr);

//    std::vector<const Tensor *> input_tensors(op->inputs()->size());
//    std::vector<Tensor *> output_tensors(op->outputs()->size());
//
//    for (int32_t j = 0; j < op->inputs()->size(); ++j)
//    {
//      const auto input_index = op->inputs()->operator[](j);
//      if (input_index != -1)
//      {
//        const auto raw_tensor = _reader->tensors()[input_index];
//
//        if (_index_to_tensor.find(raw_tensor) == _index_to_tensor.end())
//        {
//          assert(false && "Failed import operation input tensor");
//          return;
//        }
//        auto input_tensor = _index_to_tensor.at(raw_tensor).get();
//
//        input_tensors.at(j) = input_tensor;
//      }
//      else
//      {
//        input_tensors.at(j) = nullptr;
//      }
//    }
//
//    for (int32_t j = 0; j < op->outputs()->size(); ++j)
//    {
//      const auto output_index = op->outputs()->operator[](j);
//      const auto raw_tensor = _reader->tensors()[output_index];
//
//      if (_index_to_tensor.find(raw_tensor) == _index_to_tensor.end())
//      {
//        assert(false && "Failed import operation output tensor");
//        return;
//      }
//      auto output_tensor = _index_to_tensor.at(raw_tensor).get();
//      output_tensors.at(j) = output_tensor;
//    }

    const auto opcode = _reader->builtin_code(op);

    allocate(i);

    bool is_inplace = false;

    if (_inplace_op_indexes.find(i) != _inplace_op_indexes.end())
      is_inplace = true;

    kernel_executor.execute_kernel(op, opcode, this, is_inplace);

    deallocate(i);
  }

}
////
//// StaticRuntimeGraph
//StaticRuntimeGraph::StaticRuntimeGraph(IMemoryManager *memory_manager, CircleReader *circle_reader)
//  : IBaseRuntimeGraph(memory_manager, circle_reader)
//{
//}
//
//void StaticRuntimeGraph::configureGraphInputs()
//{
//
//}
//
//StaticRuntimeGraph::~StaticRuntimeGraph()
//{
////  // Release intermediate computing buffer.
////  _memory_manager->release_computing_buf();
////  _memory_manager->release_input_buf();
////  _memory_manager->release_output_buf();
//}
//
//void StaticRuntimeGraph::configure()
//{
////  // Allocate memory for intermediate computing buffer and for output buffer.
////
////  _memory_manager->allocate_computing_buf();
////  _memory_manager->allocate_output_buf();
////
////  // Set tensor's data pointer for intermediate tensors
////  for (auto &kernel : _kernels)
////  {
////    const auto output_tensors = kernel->getOutputTensors();
////
////    for (auto tensor : output_tensors)
////      _memory_manager->allocate_memory(*tensor);
////  }
////
////  // Set tensor's data pointer for output tensors
////  for (const auto output_tensor : _output_tensors)
////    _memory_manager->allocate_memory_for_output(*output_tensor);
////
////  _is_valid = true;
//}
//
//void StaticRuntimeGraph::configure_kernels()
//{
////  for (auto &kernel : _kernels)
////  {
////    kernel->configure();
////  }
//}
//
//void StaticRuntimeGraph::execute()
//{
////  if (not _is_valid)
////    configure();
////
////  for (auto &kernel : _kernels)
////  {
////    // TODO: add kernel->configure for methods with dynamic shapes
////    kernel->execute();
////  }
////
////  // Release intermediate computing buffer.
////  _memory_manager->release_computing_buf();
////
////  _is_valid = false;
//}

} // namespace luci_interpreter
