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

#include "kernels/Unpack.h"

#include "kernels/Utils.h"

//#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>

#include <stdexcept>

namespace luci_interpreter
{

namespace kernels
{

namespace
{
template <typename Scalar>
void Unpack_impl(const tflite::UnpackParams& params, const tflite::RuntimeShape& input_shape,
            const Scalar* input_data, const tflite::RuntimeShape& output_shape,
            std::vector<Scalar *> &output_datas) {
 // ruy::profiler::ScopeLabel label("Unpack");
  const int dimensions = input_shape.DimensionsCount();
  const int outputs_count = params.num_split;

  int outer_size = 1;
  int axis = params.axis;
  if (axis < 0) {
    axis += dimensions;
  }
  TFLITE_DCHECK_GE(axis, 0);
  TFLITE_DCHECK_LT(axis, dimensions);
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_shape.Dims(i);
  }
  int copy_size = 1;
  for (int i = axis + 1; i < dimensions; ++i) {
    copy_size *= input_shape.Dims(i);
  }
  TFLITE_DCHECK_EQ(output_shape.FlatSize(), copy_size * outer_size);

  for (int i = 0; i < outputs_count; ++i) {
    for (int k = 0; k < outer_size; k++) {
      Scalar* output_ptr = output_datas[i] + copy_size * k;
      int loc = k * outputs_count * copy_size + i * copy_size;
      memcpy(output_ptr, input_data + loc, copy_size * sizeof(Scalar));
    }
  }
}
}

Unpack::Unpack(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs)
  : Kernel(std::move(inputs), std::move(outputs))
{
}

void Unpack::configure(luci::CircleReader *circle_reader, int32_t index)
{
//  const Shape &input_shape = input()->shape();
//
//  int axis = _params.axis;
//  if (axis < 0)
//    axis += input()->shape().num_dims();
//  assert(axis >= 0 && axis < input_shape.num_dims());
//
//  Shape output_shape(input_shape.num_dims() - 1);
//  int out_index = 0;
//  for (int in_index = 0; in_index < input_shape.num_dims(); ++in_index)
//  {
//    if (in_index != axis)
//      output_shape.dim(out_index++) = input_shape.dim(in_index);
//  }
//
//  for (Tensor *output : _outputs)
//  {
//    assert(output->element_type() == input()->element_type());
//    output->resize(output_shape);
//  }
}

template <typename T> void Unpack::executeImpl(luci::CircleReader *circle_reader, int32_t index) const
{
  circle::OperatorT oper_t;
  circle_reader->operators()[index]->UnPackTo(&oper_t);
  const auto *options = oper_t.builtin_options.AsUnpackOptions();

  tflite::UnpackParams params{};
  params.axis = options->axis;
  params.num_split = _outputs.size();


  const auto input_tensor_shape = luci::wrap(circle_reader->tensors()[input_ind()]->shape());
  tflite::RuntimeShape input_runtime_shape(input_tensor_shape.size());
  for (int i = 0; i < input_tensor_shape.size(); ++i)
  {
    input_runtime_shape.SetDim(i, input_tensor_shape[i]);
  }

  const auto output_tensor_shape = luci::wrap(circle_reader->tensors()[output_ind(0)]->shape());
  tflite::RuntimeShape output_runtime_shape(output_tensor_shape.size());
  for (int i = 0; i < output_tensor_shape.size(); ++i)
  {
    output_runtime_shape.SetDim(i, output_tensor_shape[i]);
  }

  std::vector<T *> all_data;

  for (uint32_t i = 0; i < _outputs.size(); ++i)
  {
    all_data.push_back(getTensorData<T>(_outputs.at(i).first));
  }


  Unpack_impl<T>(params, input_runtime_shape, getTensorData<T>(input()),
                                   output_runtime_shape, all_data);
}

void Unpack::execute(luci::CircleReader *circle_reader, int32_t index) const
{
  auto input_dtype = luci::luci_datatype(circle_reader->tensors()[input_ind()]->type());

  switch (input_dtype)
  {
    case DataType::FLOAT32:
      return executeImpl<float>(circle_reader, index);
    case DataType::U8:
      return executeImpl<uint8_t>(circle_reader, index);
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
