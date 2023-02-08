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

#ifndef LUCI_INTERPRETER_CORE_RUNTIMEGRAPH_H
#define LUCI_INTERPRETER_CORE_RUNTIMEGRAPH_H

#include "luci_interpreter/core/Tensor.h"
#include "memory_managers/MemoryManager.h"
//#include "core/Kernel.h"
#include "luci_interpreter/core/reader/CircleMicroReader.h"

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>

namespace luci_interpreter
{

class RuntimeModule;

class IBaseRuntimeGraph
{
public:
  explicit IBaseRuntimeGraph(IMemoryManager *memory_manager, CircleReader *circle_reader);
  virtual ~IBaseRuntimeGraph() = default;

  Tensor *addTensor(const circle::Tensor *raw_tensor, std::unique_ptr<Tensor> &&tensor);
//  void shrink_to_fit_tensors()
//  {
//    _tensors.shrink_to_fit();
//  }

//  void resize_input_tensors(int32_t n)
//  {
//    _input_tensors.resize(n);
//  }
//
//  void resize_output_tensors(int32_t n)
//  {
//    _output_tensors.resize(n);
//  }


#ifndef DIS_QUANT
  AffineQuantization *addAffineQuantization(std::unique_ptr<AffineQuantization> &&quantization);

  void addIntermediateTensorAffineQuantization(AffineQuantization *intermediate_tensor);

  const std::vector<AffineQuantization *> &getIntermediateAffineQuantizations() const
  {
    return _intermediate_tensors_affine_quantizations;
  }
#endif
  //void addInputTensor(Tensor *input_tensor, int pos);
  //void addOutputTensor(Tensor *output_tensor, int pos);

  //void configureAllocations(Tensor *tensor);

  std::unordered_map<const circle::Tensor *, uint8_t *> *getIndexToTensor()
  {
    return &_index_to_tensor;
  }

  const circle::Tensor *getCircleTensorByIndex(int32_t index);

  uint8_t *getDataByCircleTensor(const circle::Tensor *raw_tensor);

  void setNullDataByCircleTensor(const circle::Tensor *raw_tensor, const circle::Tensor *output_tensor);

  uint8_t *getDataBufferFromCircleTensor(const circle::Tensor *raw_tensor);

  virtual void configureGraphInputs() = 0;

  void addInplaceOpIndex(uint32_t index)
  {
    _inplace_op_indexes.insert(index);
  }

  std::vector<const circle::Tensor *> getInputTensors() const; //{ return _input_tensors; }
  std::vector<const circle::Tensor *> getOutputTensors() const; //{ return _output_tensors; }

  //void addKernel(std::unique_ptr<Kernel> &&kernel);

  virtual void execute() = 0;
  virtual void configure() = 0;

  virtual void configure_kernels() = 0;

  void invalidate() { _is_valid = false; }
  bool isValid() const { return _is_valid; }

protected:
  IMemoryManager *_memory_manager;
  //std::vector<std::unique_ptr<Tensor>> _tensors;
  std::unordered_map<const circle::Tensor *, uint8_t *> _index_to_tensor;


#ifndef DIS_QUANT
  std::vector<std::unique_ptr<AffineQuantization>> _affine_quantizations;
  std::vector<AffineQuantization *> _intermediate_tensors_affine_quantizations;
#endif
  //std::vector<Tensor *> _input_tensors;

  //std::vector<Tensor *> _output_tensors;

  bool _is_valid = false;

  CircleReader *_reader;
  std::unordered_set<uint32_t> _inplace_op_indexes;

  // Kernels in execution order.
  //std::vector<std::unique_ptr<Kernel>> _kernels;
};

class RuntimeGraph final : public IBaseRuntimeGraph
{
public:
  explicit RuntimeGraph(IMemoryManager *memory_manager, CircleReader *circle_reader);
  ~RuntimeGraph() final;

  void execute() final;
  void configure() final;

  void configureGraphInputs() final;

  void configure_kernels() final { assert(false && "Use it only with static allocations"); }

private:
  void buildAllocDeallocPlan();
  void allocate(size_t kernel_index);
  void deallocate(size_t kernel_index);

private:
  // Tensors that are not used anymore after given op
  std::vector<std::vector<const circle::Tensor *>> _alloc_plan;
  std::vector<std::vector<const circle::Tensor *>> _dealloc_plan;
};

//class StaticRuntimeGraph final : public IBaseRuntimeGraph
//{
//public:
//  explicit StaticRuntimeGraph(IMemoryManager *memory_manager, CircleReader *circle_reader);
//  ~StaticRuntimeGraph() final;
//
//  void configureGraphInputs() final;
//  void execute() final;
//  void configure() final;
//
//  void configure_kernels() final;
//};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_CORE_RUNTIMEGRAPH_H
