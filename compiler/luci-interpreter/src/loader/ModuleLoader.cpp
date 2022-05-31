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
std::unique_ptr<Kernel> build_kernel_micro_CircleFullyConnected(std::vector<std::pair<const Tensor *, int32_t>> &inputs, std::vector<std::pair<Tensor *, int32_t>> &output)
{
  return std::make_unique<kernels::FullyConnected>(std::move(inputs), std::move(output));
}

std::unique_ptr<Kernel> build_kernel_micro_CircleLogistic(std::vector<std::pair<const Tensor *, int32_t>> &inputs, std::vector<std::pair<Tensor *, int32_t>> &output)
{
  return std::make_unique<kernels::Logistic>(std::move(inputs), std::move(output));
}

std::unique_ptr<Kernel> build_kernel_micro_CircleExpandDims(std::vector<std::pair<const Tensor *, int32_t>> &inputs, std::vector<std::pair<Tensor *, int32_t>> &output)
{
  return std::make_unique<kernels::ExpandDims>(std::move(inputs), std::move(output));
}

std::unique_ptr<Kernel> build_kernel_micro_CircleConv2d(std::vector<std::pair<const Tensor *, int32_t>> &inputs, std::vector<std::pair<Tensor *, int32_t>> &output, RuntimeGraph *runtime_graph)
{
  auto scratchpad = std::make_unique<Tensor>(0);
  scratchpad->set_data_buffer(nullptr);
  scratchpad->set_allocatable(false);

  Tensor *tmp = runtime_graph->addTensor(std::move(scratchpad));
  output.emplace_back(tmp, -1);

  return std::make_unique<kernels::Conv2D>(std::move(inputs), std::move(output));
}

std::unique_ptr<Kernel> build_kernel_micro_CircleReshape(std::vector<std::pair<const Tensor *, int32_t>> &inputs, std::vector<std::pair<Tensor *, int32_t>> &output)
{
  return std::make_unique<kernels::Reshape>(std::move(inputs), std::move(output));
}

std::unique_ptr<Kernel> build_kernel_micro_CircleMaxPool2D(std::vector<std::pair<const Tensor *, int32_t>> &inputs, std::vector<std::pair<Tensor *, int32_t>> &output)
{
  return std::make_unique<kernels::MaxPool2D>(std::move(inputs), std::move(output));
}

std::unique_ptr<Kernel> build_kernel_micro_CirclePad(std::vector<std::pair<const Tensor *, int32_t>> &inputs, std::vector<std::pair<Tensor *, int32_t>> &output)
{
  return std::make_unique<kernels::Pad>(std::move(inputs), std::move(output));
}

std::unique_ptr<Kernel> build_kernel_micro_CircleSpaceToBatchNd(std::vector<std::pair<const Tensor *, int32_t>> &inputs, std::vector<std::pair<Tensor *, int32_t>> &output)
{
  return std::make_unique<kernels::SpaceToBatchND>(std::move(inputs), std::move(output));
}

std::unique_ptr<Kernel> build_kernel_micro_CircleBatchToSpaceNd(std::vector<std::pair<const Tensor *, int32_t>> &inputs, std::vector<std::pair<Tensor *, int32_t>> &output)
{
  return std::make_unique<kernels::BatchToSpaceND>(std::move(inputs), std::move(output));
}

std::unique_ptr<Kernel> build_kernel_micro_CircleAdd(std::vector<std::pair<const Tensor *, int32_t>> &inputs, std::vector<std::pair<Tensor *, int32_t>> &output)
{
  return std::make_unique<kernels::Add>(std::move(inputs), std::move(output));
}

std::unique_ptr<Kernel> build_kernel_micro_CircleConcatenation(std::vector<std::pair<const Tensor *, int32_t>> &inputs, std::vector<std::pair<Tensor *, int32_t>> &output)
{
  return std::make_unique<kernels::Concatenation>(std::move(inputs), std::move(output));
}


} // namespace

void ModuleLoader::load()
{
 // printf("\nload\n");
  const circle::Model *model = circle::GetModel(_model_data_raw);

  luci::CircleReader *reader = _runtime_module->get_circle_reader();
//  printf("\newade\n");

  //_circle_reader = std::make_unique<luci::CircleReader>();
  if (!reader->parse(model))
    throw std::runtime_error("Error during parse");

 // printf("\nparse\n");
  for (uint32_t g = 0; g < reader->num_subgraph(); ++g)
  {
    _graph_to_runtime_graph.emplace_back(_runtime_module->addGraph(_memory_manager));

    if (!reader->select_subgraph(g))
      throw std::runtime_error("Error during select subgraph");

    assert(!reader->tensors().null());

    for (const auto input : reader->inputs())
    {
     // printf("\ninput\n");
      const auto tensor = reader->tensors()[input];

     // printf("\n1\n");
     //  Create dtype
      const auto dtype = luci::luci_datatype(tensor->type());
     // printf("\n2\n");
      // Create Shape
      const auto tensor_shape = luci::wrap(tensor->shape());
     // printf("\n3\n");
      //const auto dims = tensor_shape; // in NHWC
      //Shape shape(static_cast<int>(dims.size()));
      auto size = 1;
      for (uint32_t r = 0; r < tensor_shape.size(); ++r)
      {
        //shape.dim(r) = dims[r];
        size *= tensor_shape[r];
      }
     // printf("\n4\n");
      size *= getDataTypeSize(dtype);
     // printf("\n5\n");
      // Create quantization;
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

      //auto tensor_interpreter = std::make_unique<Tensor>(dtype, std::move(shape), nullptr);
      auto tensor_interpreter = std::make_unique<Tensor>(size);

      _memory_manager->allocate_memory(*tensor_interpreter);
     // printf("\n7\n");
      _graph_to_runtime_graph.back()->add_input_tensor(tensor_interpreter.get());
     // printf("\n8\n");
      _index_to_tensor.emplace(input, tensor_interpreter.get());
    //  printf("\n9\n");
      //inputs_index_tensor.at(ind) = input;
      //ind++;

      _graph_to_runtime_graph.back()->addTensor(std::move(tensor_interpreter));
   //   printf("\nafter input\n");
    }

    for (uint32_t i = 0; i < reader->tensors().size(); ++i)
    {
   //   printf("\nten i = %d\n", i);
      auto const const_tensor = reader->tensors()[i];
      if (const_tensor->is_variable())
        continue;

      auto const buffer = luci::wrap(reader->buffers()[const_tensor->buffer()]->data());
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


      //  Create dtype
      const auto dtype = luci::luci_datatype(const_tensor->type());

      // Create Shape
      const auto tensor_shape = luci::wrap(const_tensor->shape());

      //const auto dims = tensor_shape; // in NHWC
      //Shape shape(static_cast<int>(dims.size()));
      auto size = 1;
      for (uint32_t r = 0; r < tensor_shape.size(); ++r)
      {
        //shape.dim(r) = dims[r];
        size *= tensor_shape[r];
      }

      size *= getDataTypeSize(dtype);

      auto tensor_interpreter = std::make_unique<Tensor>(size);

      //_memory_manager->allocate_memory(*tensor_interpreter);

      tensor_interpreter->write_data_without_copy(static_cast<void *>(const_cast<unsigned char *>(buffer.data())));

      _index_to_tensor.emplace(i, tensor_interpreter.get());

      _graph_to_runtime_graph.back()->addTensor(std::move(tensor_interpreter));

    }

    for (uint32_t i = 0; i < reader->operators().size(); ++i)
    {
     // printf("\nSize of kernel = %lu\n", sizeof(Kernel));
     // printf("\nSize of kernel params for conv2d = %lu\n", sizeof(KernelWithParams<Conv2DParams>));
      const auto op = reader->operators()[i];
      assert(op != nullptr);

      circle::OperatorT oper_t;
      op->UnPackTo(&oper_t);
   //   printf("\nop i = %d\n", i);

     // std::vector<int32_t> &inputs = oper_t.inputs;
     // std::vector<int32_t> &outputs = oper_t.outputs;

     // const auto opcodes = reader.opcodes();

      std::vector<std::pair<const Tensor *, int32_t>> input_tensors;
      for (const int32_t input_tensor_index : oper_t.inputs)
      {
        if (_index_to_tensor.find(input_tensor_index) != _index_to_tensor.end())
          input_tensors.emplace_back(const_cast<const Tensor *>(_index_to_tensor.at(input_tensor_index)), input_tensor_index);
      }


      /*
       * Output tensor for current Op
       */
      auto output_tensor = reader->tensors()[oper_t.outputs[0]];

      circle::BuiltinOperator builtincode = reader->builtin_code(op);

      Tensor *output = nullptr;

      if (std::find(reader->inputs().begin(), reader->inputs().end(), oper_t.inputs.at(0)) == reader->inputs().end() and
          (builtincode == circle::BuiltinOperator_LOGISTIC or builtincode == circle::BuiltinOperator_EXPAND_DIMS or builtincode == circle::BuiltinOperator_RESHAPE))
      {
        auto input_tensor = input_tensors.at(0).first;
        output = const_cast<Tensor *>(input_tensor);
      } else
      {

        //  Create dtype
        const auto dtype = luci::luci_datatype(output_tensor->type());

        // Create Shape
        const auto tensor_shape = luci::wrap(output_tensor->shape());

        //const auto dims = tensor_shape; // in NHWC
        //Shape shape(static_cast<int>(dims.size()));
        auto size = 1;
        for (uint32_t r = 0; r < tensor_shape.size(); ++r)
        {
          //shape.dim(r) = dims[r];
          size *= tensor_shape[r];
        }

        size *= getDataTypeSize(dtype);

        auto tensor_interpreter = std::make_unique<Tensor>(size);

        output = tensor_interpreter.get();

        _graph_to_runtime_graph.back()->addTensor(std::move(tensor_interpreter));
      }

      _index_to_tensor.emplace(oper_t.outputs[0], output);

      if (i == reader->operators().size() - 1)
        _graph_to_runtime_graph.back()->add_output_tensor(output);

      std::vector<std::pair<Tensor *, int32_t>> output_tensors;
      output_tensors.emplace_back(output, oper_t.outputs[0]);

      switch (builtincode)
      {
        case circle::BuiltinOperator_FULLY_CONNECTED:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleFullyConnected(input_tensors, output_tensors);
          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_LOGISTIC:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleLogistic(input_tensors, output_tensors);
          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_EXPAND_DIMS:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleExpandDims(input_tensors, output_tensors);
          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_CONV_2D:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleConv2d(input_tensors, output_tensors, _graph_to_runtime_graph.back());
          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_RESHAPE:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleReshape(input_tensors, output_tensors);
          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_MAX_POOL_2D:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleMaxPool2D(input_tensors, output_tensors);
          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_PAD:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CirclePad(input_tensors, output_tensors);
          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_SPACE_TO_BATCH_ND:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleSpaceToBatchNd(input_tensors, output_tensors);
          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_BATCH_TO_SPACE_ND:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleBatchToSpaceNd(input_tensors, output_tensors);
          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_ADD:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleAdd(input_tensors, output_tensors);
          _graph_to_runtime_graph.back()->addKernel(std::move(kernel));
          break;
        }
        case circle::BuiltinOperator_CONCATENATION:
        {
          std::unique_ptr<Kernel> kernel = build_kernel_micro_CircleConcatenation(input_tensors, output_tensors);
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

}

} // namespace luci_interpreter
