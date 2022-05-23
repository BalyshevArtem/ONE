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
#include "luci_interpreter/core/KernelParams.h"
#include "kernels/FullyConnected.h"
#include "kernels/Logistic.h"
#include "kernels/ExpandDims.h"
#include "kernels/Conv2D.h"
#include "kernels/Reshape.h"
#include "kernels/MaxPool2D.h"
#include "kernels/Pad.h"
#include "kernels/SpaceToBatchND.h"
#include "kernels/BatchToSpaceND.h"
#include "kernels/Add.h"
#include "kernels/Concatenation.h"

//#include "../../../luci-micro/mbed-os/mio/circle/schema_generated.h"
//#include "../../../luci/import/include/luci/Import/CircleReader.h"

#include "CircleReader.h"
//#include "../../../luci/import/include/luci/Import/GraphBuilderContext.h"

namespace luci_interpreter
{

//ModuleLoader::ModuleLoader(const luci::Module *module, RuntimeModule *runtime_module,
//                           RuntimeToIR &runtime_to_ir,
//                           std::unordered_map<const loco::Node *, Tensor *> &node_to_tensor,
//                           IMemoryManager *memory_manager)
//  : _module(module), _runtime_module(runtime_module), _runtime_to_ir(runtime_to_ir),
//    _node_to_tensor(node_to_tensor), _memory_manager(memory_manager)
//{
//}

ModuleLoader::ModuleLoader(char *model_data_raw, RuntimeModule *runtime_module,
                           IMemoryManager *memory_manager)
  : _model_data_raw(model_data_raw), _runtime_module(runtime_module),
   _memory_manager(memory_manager)
{
}

namespace
{
std::unique_ptr<Kernel> build_kernel_micro_CircleFullyConnected(const circle::OperatorT &op, std::vector<Tensor *> &inputs, Tensor *output)
{
  const Tensor *input = inputs[0];
  const Tensor *weights = inputs[1];
  const Tensor *bias = nullptr;
  if (inputs.size() == 3)
    bias = inputs[2];

  FullyConnectedParams params{};

  const auto *options = op.builtin_options.AsFullyConnectedOptions();
  params.activation= luci::luci_actfunc(options->fused_activation_function);
  params.keep_num_dims = options->keep_num_dims;

  return std::make_unique<kernels::FullyConnected>(input, weights, bias, output, params);
}

std::unique_ptr<Kernel> build_kernel_micro_CircleLogistic(const circle::OperatorT &op, std::vector<Tensor *> &inputs, Tensor *output)
{
  return std::make_unique<kernels::Logistic>(inputs[0], output);
}

std::unique_ptr<Kernel> build_kernel_micro_CircleExpandDims(const circle::OperatorT &op, std::vector<Tensor *> &inputs, Tensor *output)
{
  const Tensor *input = inputs[0];
  const Tensor *axis = inputs[1];

  return std::make_unique<kernels::ExpandDims>(input, axis, output);
}

std::unique_ptr<Kernel> build_kernel_micro_CircleConv2d(const circle::OperatorT &op, std::vector<Tensor *> &inputs, Tensor *output, RuntimeGraph *runtime_graph)
{
  const Tensor *input = inputs[0];
  const Tensor *filter = inputs[1];
  const Tensor *bias = nullptr;
  if (inputs.size() == 3)
    bias = inputs[2];

  auto scratchpad = std::make_unique<Tensor>(DataType::U8, Shape({}), nullptr);
  scratchpad->set_observable(false);
  scratchpad->set_data_buffer(nullptr);

  Tensor *tmp = runtime_graph->addTensor(std::move(scratchpad));

  Conv2DParams params{};

  const auto *options = op.builtin_options.AsConv2DOptions();
  params.padding = luci::luci_padding(options->padding);
  params.stride_height = options->stride_h;
  params.stride_width = options->stride_w;
  params.dilation_height_factor = options->dilation_h_factor;
  params.dilation_width_factor = options->dilation_w_factor;
  params.activation = luci::luci_actfunc(options->fused_activation_function);

  return std::make_unique<kernels::Conv2D>(input, filter, bias, output, tmp, params);
}

std::unique_ptr<Kernel> build_kernel_micro_CircleReshape(const circle::OperatorT &op, std::vector<Tensor *> &inputs, Tensor *output)
{
  const Tensor *input = inputs[0];
  const Tensor *shape = inputs[1];

  return std::make_unique<kernels::Reshape>(input, shape, output);
}

std::unique_ptr<Kernel> build_kernel_micro_CircleMaxPool2D(const circle::OperatorT &op, std::vector<Tensor *> &inputs, Tensor *output)
{
  const Tensor *input = inputs[0];

  Pool2DParams params{};

  const auto *options = op.builtin_options.AsPool2DOptions();
  params.padding = luci::luci_padding(options->padding);
  params.filter_height = options->filter_height;
  params.filter_width = options->filter_width;
  params.stride_height = options->stride_h;
  params.stride_width = options->stride_w;
  params.activation= luci::luci_actfunc(options->fused_activation_function);

  return std::make_unique<kernels::MaxPool2D>(input, output, params);
}

std::unique_ptr<Kernel> build_kernel_micro_CirclePad(const circle::OperatorT &op, std::vector<Tensor *> &inputs, Tensor *output)
{
  const Tensor *input = inputs[0];
  const Tensor *paddings = inputs[1];

  return std::make_unique<kernels::Pad>(input, paddings, output);
}

std::unique_ptr<Kernel> build_kernel_micro_CircleSpaceToBatchNd(const circle::OperatorT &op, std::vector<Tensor *> &inputs, Tensor *output, const Shape shape)
{
  const Tensor *input = inputs[0];
  const Tensor *block_shape = inputs[1];
  const Tensor *paddings = inputs[2];

  return std::make_unique<kernels::SpaceToBatchND>(input, block_shape, paddings, output, shape);
}

std::unique_ptr<Kernel> build_kernel_micro_CircleBatchToSpaceNd(const circle::OperatorT &op, std::vector<Tensor *> &inputs, Tensor *output, const Shape shape)
{
  const Tensor *input = inputs[0];
  const Tensor *block_shape = inputs[1];
  const Tensor *crops = inputs[2];

  return std::make_unique<kernels::BatchToSpaceND>(input, block_shape, crops, output, shape);
}

std::unique_ptr<Kernel> build_kernel_micro_CircleAdd(const circle::OperatorT &op, std::vector<Tensor *> &inputs, Tensor *output)
{
  const Tensor *input_1 = inputs[0];
  const Tensor *input_2 = inputs[1];

  AddParams params{};

  const auto *options = op.builtin_options.AsAddOptions();
  params.activation = luci::luci_actfunc(options->fused_activation_function);

  return std::make_unique<kernels::Add>(input_1, input_2, output, params);
}

std::unique_ptr<Kernel> build_kernel_micro_CircleConcatenation(const circle::OperatorT &op, std::vector<Tensor *> &inputs, Tensor *output)
{
  std::vector<const Tensor *> tensors_inputs(inputs.size());

  for (uint32_t i = 0; i < inputs.size(); ++i)
  {
    tensors_inputs[i] = inputs[i];
  }

  ConcatenationParams params{};

  const auto *options = op.builtin_options.AsConcatenationOptions();
  params.activation = luci::luci_actfunc(options->fused_activation_function);
  params.axis = options->axis;

  return std::make_unique<kernels::Concatenation>(std::move(tensors_inputs), output, params);
}


} // namespace

void ModuleLoader::load()
{
  const circle::Model *model = circle::GetModel(_model_data_raw);

  luci::CircleReader reader;
  if (!reader.parse(model))
    throw std::runtime_error("Error during parse");

  for (uint32_t g = 0; g < reader.num_subgraph(); ++g)
  {
    _graph_to_runtime_graph.emplace_back(_runtime_module->addGraph(_memory_manager));

    if (!reader.select_subgraph(g))
      throw std::runtime_error("Error during select subgraph");

  //  const auto operators = reader.operators();
  //  const auto tensors = reader.tensors();
    assert(!reader.tensors().null());

    //auto circle_metadata = std::make_unique<luci::CircleImportMetadata>(reader);

//    auto nodefinder = std::make_unique<luci::IndexNodeFinder>();
//    auto tensoroutputs = std::make_unique<luci::IndexTensorOutputs>();
//
//    for (uint32_t i = 0; i < operators.size(); ++i)
//    {
//      const auto op = operators[i];
//      assert(op != nullptr);
//      const auto outputs = luci::wrap(op->outputs());
//
//      for (uint32_t j = 0; j < outputs.size(); ++j)
//      {
//        auto tidx = outputs[j];
//        tensoroutputs->enroll(tidx);
//      }
//    }

    //auto result = reader.inputs();
   // std::vector<int> inputs_index_tensor(reader.inputs().size());
   // int ind = 0;
    for (const auto input : reader.inputs())
    {
      const auto tensor = reader.tensors()[input];

      // Create dtype
      const auto dtype = luci::luci_datatype(tensor->type());

      // Create Shape
      const auto tensor_shape = luci::wrap(tensor->shape());

      const auto dims = tensor_shape; // in NHWC
      Shape shape(static_cast<int>(dims.size()));
      for (uint32_t r = 0; r < dims.size(); ++r)
      {
        shape.dim(r) = dims[r];
      }

      // Create quantization;
      AffineQuantization quantization;
      const auto quantization_tensor = tensor->quantization();
      if (quantization_tensor != nullptr)
      {
        auto quantparam = luci::luci_quantparam(quantization_tensor);
        if (quantparam)
        {
          quantization.scale.assign(quantparam->scale.cbegin(), quantparam->scale.cend());
          quantization.zero_point.assign(quantparam->zerop.cbegin(), quantparam->zerop.cend());
          quantization.quantized_dimension = quantparam->quantized_dimension;
        }
      }

      auto tensor_interpreter = std::make_unique<Tensor>(dtype, std::move(shape), nullptr);

      _memory_manager->allocate_memory(*tensor_interpreter);

      _graph_to_runtime_graph.back()->add_input_tensor(tensor_interpreter.get());

      _index_to_tensor.emplace(input, tensor_interpreter.get());

      //inputs_index_tensor.at(ind) = input;
      //ind++;

      _graph_to_runtime_graph.back()->addTensor(std::move(tensor_interpreter));
    }

    for (uint32_t i = 0; i < reader.tensors().size(); ++i)
    {
      auto const const_tensor = reader.tensors()[i];
      if (const_tensor->is_variable())
        continue;

      auto const buffer = luci::wrap(reader.buffers()[const_tensor->buffer()]->data());
      auto const const_dims = luci::wrap(const_tensor->shape()); // in NHWC
      if (const_dims.empty() && buffer.empty())
      {
        // unknown shape tensor and scalar tensor
        continue;
      }

      uint32_t num_elements = 1;
      for (uint32_t r = 0; r < const_dims.size(); ++r)
      {
        num_elements = num_elements * const_dims[r];
      }

      if (buffer.empty() && num_elements > 0)
      {
        // normal empty tensor
        continue;
      }

      // Create dtype
      const auto dtype = luci::luci_datatype(const_tensor->type());

      // Create Shape
      const auto tensor_shape = luci::wrap(const_tensor->shape());

      const auto dims = tensor_shape; // in NHWC
      Shape shape(static_cast<int>(dims.size()));
      for (uint32_t r = 0; r < dims.size(); ++r)
      {
        shape.dim(r) = dims[r];
      }

      // Create quantization;
      AffineQuantization quantization;
      const auto quantization_tensor = const_tensor->quantization();
      if (quantization_tensor != nullptr)
      {
        auto quantparam = luci::luci_quantparam(quantization_tensor);
        if (quantparam)
        {
          quantization.scale.assign(quantparam->scale.cbegin(), quantparam->scale.cend());
          quantization.zero_point.assign(quantparam->zerop.cbegin(), quantparam->zerop.cend());
          quantization.quantized_dimension = quantparam->quantized_dimension;
        }
      }

      //printf("\n Tensor size = %lu\n", sizeof(Tensor));

      auto tensor_interpreter = std::make_unique<Tensor>(dtype, std::move(shape), nullptr);

      //_memory_manager->allocate_memory(*tensor_interpreter);

      tensor_interpreter->write_data_without_copy(static_cast<void *>(const_cast<unsigned char *>(buffer.data())));

      _index_to_tensor.emplace(i, tensor_interpreter.get());

      _graph_to_runtime_graph.back()->addTensor(std::move(tensor_interpreter));

    }

    for (uint32_t i = 0; i < reader.operators().size(); ++i)
    {
     // printf("\nSize of kernel = %lu\n", sizeof(Kernel));
     // printf("\nSize of kernel params for conv2d = %lu\n", sizeof(KernelWithParams<Conv2DParams>));
      const auto op = reader.operators()[i];
      assert(op != nullptr);

      circle::OperatorT oper_t;
      op->UnPackTo(&oper_t);

     // std::vector<int32_t> &inputs = oper_t.inputs;
     // std::vector<int32_t> &outputs = oper_t.outputs;

     // const auto opcodes = reader.opcodes();

      std::vector<Tensor *> input_tensors;
      for (const int32_t input_tensor_index : oper_t.inputs)
      {
        if (_index_to_tensor.find(input_tensor_index) != _index_to_tensor.end())
          input_tensors.push_back(_index_to_tensor.at(input_tensor_index));
      }


      /*
       * Output tensor for current Op
       */
      auto output_tensor = reader.tensors()[oper_t.outputs[0]];

      // Create dtype
      const auto dtype = luci::luci_datatype(output_tensor->type());

      // Create Shape
      const auto tensor_shape = luci::wrap(output_tensor->shape());

      const auto dims = tensor_shape; // in NHWC
      Shape shape(static_cast<int>(dims.size()));
      for (uint32_t r = 0; r < dims.size(); ++r)
      {
        shape.dim(r) = dims[r];
      }

      // Create quantization;
      AffineQuantization quantization;
      const auto quantization_tensor = output_tensor->quantization();
      if (quantization_tensor != nullptr)
      {
        auto quantparam = luci::luci_quantparam(quantization_tensor);
        if (quantparam)
        {
          quantization.scale.assign(quantparam->scale.cbegin(), quantparam->scale.cend());
          quantization.zero_point.assign(quantparam->zerop.cbegin(), quantparam->zerop.cend());
          quantization.quantized_dimension = quantparam->quantized_dimension;
        }
      }

      circle::BuiltinOperator builtincode = reader.builtin_code(op);

      Tensor *output = nullptr;

      if (std::find(reader.inputs().begin(), reader.inputs().end(), oper_t.inputs.at(0)) == reader.inputs().end() and
          (builtincode == circle::BuiltinOperator_LOGISTIC or builtincode == circle::BuiltinOperator_EXPAND_DIMS or builtincode == circle::BuiltinOperator_RESHAPE))
      {
        auto input_tensor = input_tensors.at(0);
        output = input_tensor;
      } else
      {
        auto tensor_interpreter = std::make_unique<Tensor>(dtype, (shape), nullptr);
        output = tensor_interpreter.get();

        _graph_to_runtime_graph.back()->addTensor(std::move(tensor_interpreter));
      }



      //_memory_manager->allocate_memory(*tensor_interpreter);

     // _graph_to_runtime_graph.back()->add_input_tensor(tensor_interpreter.get());

      _index_to_tensor.emplace(oper_t.outputs[0], output);

      if (i == reader.operators().size() - 1)
        _graph_to_runtime_graph.back()->add_output_tensor(output);

      switch (builtincode)
      {
        case circle::BuiltinOperator_FULLY_CONNECTED:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleFullyConnected(oper_t, input_tensors, output);

         // _kernel_to_tensor.emplace(kernel.get(), output);

          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_LOGISTIC:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleLogistic(oper_t, input_tensors, output);

         // _kernel_to_tensor.emplace(kernel.get(), output);

          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_EXPAND_DIMS:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleExpandDims(oper_t, input_tensors, output);

         // _kernel_to_tensor.emplace(kernel.get(), output);

          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_CONV_2D:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleConv2d(oper_t, input_tensors, output, _graph_to_runtime_graph.back());

         // _kernel_to_tensor.emplace(kernel.get(), output);

          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_RESHAPE:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleReshape(oper_t, input_tensors, output);

         // _kernel_to_tensor.emplace(kernel.get(), output);

          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_MAX_POOL_2D:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleMaxPool2D(oper_t, input_tensors, output);

          //_kernel_to_tensor.emplace(kernel.get(), output);

          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_PAD:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CirclePad(oper_t, input_tensors, output);

         // _kernel_to_tensor.emplace(kernel.get(), output);

          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_SPACE_TO_BATCH_ND:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleSpaceToBatchNd(oper_t, input_tensors, output, shape);

         // _kernel_to_tensor.emplace(kernel.get(), output);

          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_BATCH_TO_SPACE_ND:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleBatchToSpaceNd(oper_t, input_tensors, output, shape);

          //_kernel_to_tensor.emplace(kernel.get(), output);

          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_ADD:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleAdd(oper_t, input_tensors, output);

         // _kernel_to_tensor.emplace(kernel.get(), output);

          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_CONCATENATION:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleConcatenation(oper_t, input_tensors, output);

          // _kernel_to_tensor.emplace(kernel.get(), output);

          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        default:
          throw std::runtime_error("not supported kernel");
      }
    }

    _graph_to_runtime_graph.back()->make_tensor_alloca_plan();
    //_graph_to_runtime_graph.back()->make_tensor_configure();

//    // graph outputs
//    for (auto output : reader.outputs())
//    {
//      const auto tensor = tensors[output];
//
//      // Create dtype
//      const auto dtype = luci::luci_datatype(tensor->type());
//
//      // Create Shape
//      const auto tensor_shape = luci::wrap(tensor->shape());
//
//      const auto dims = tensor_shape; // in NHWC
//      Shape shape(static_cast<int>(dims.size()));
//      for (uint32_t r = 0; r < dims.size(); ++r)
//      {
//        shape.dim(r) = dims[r];
//      }
//
//      // Create quantization;
//      AffineQuantization quantization;
//      const auto quantization_tensor = tensor->quantization();
//      if (quantization_tensor != nullptr)
//      {
//        auto quantparam = luci::luci_quantparam(quantization_tensor);
//        if (quantparam)
//        {
//          quantization.scale.assign(quantparam->scale.cbegin(), quantparam->scale.cend());
//          quantization.zero_point.assign(quantparam->zerop.cbegin(), quantparam->zerop.cend());
//          quantization.quantized_dimension = quantparam->quantized_dimension;
//        }
//      }
//
//      auto tensor_interpreter = std::make_unique<Tensor>(dtype, std::move(shape), std::move(quantization),
//                                                         luci::tensor_name(tensor));
//
//     // _memory_manager->allocate_memory(*tensor_interpreter);
//
//      _graph_to_runtime_graph.back()->add_output_tensor(tensor_interpreter.get());
//
//      _index_to_tensor.emplace(output, tensor_interpreter.get());
//
//      _graph_to_runtime_graph.back()->addTensor(std::move(tensor_interpreter));
//    }
  }


  // Runtime graphs have to be created in advance, because they will be needed during the loading
  // process for control flow nodes.
//  for (size_t i = 0; i < _module->size(); ++i)
//  {
//    _graph_to_runtime_graph.emplace(_module->graph(i), _runtime_module->addGraph(_memory_manager));
//  }
//  for (size_t i = 0; i < _module->size(); ++i)
//  {
//    const loco::Graph *graph = _module->graph(i);
//    RuntimeGraph *runtime_graph = _graph_to_runtime_graph.at(graph);
//    GraphLoader loader(graph, runtime_graph, _runtime_to_ir, _graph_to_runtime_graph,
//                       _node_to_tensor, _memory_manager);
//    loader.loadTensors();
//    loader.initInputOutputTensors();
//    loader.loadOperators();
//  }
}

} // namespace luci_interpreter
