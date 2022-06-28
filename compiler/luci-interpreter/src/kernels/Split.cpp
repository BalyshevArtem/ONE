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

#include "Split.h"

#include "Utils.h"

//#include <tensorflow/lite/kernels/internal/reference/reference_ops.h>
//#include <tensorflow/lite/kernels/internal/optimized/optimized_ops.h>

//#include "PALSplit.h"

namespace luci_interpreter
{
namespace kernels
{

namespace
{
template <typename Scalar>
void Split_impl(const tflite::SplitParams& params, const tflite::RuntimeShape& input_shape,
           const Scalar* input_data, std::vector<tflite::RuntimeShape> &output_shapes,
           std::vector<Scalar *> &output_data) {
  //ruy::profiler::ScopeLabel label("Split");
  const int split_dimensions = input_shape.DimensionsCount();
  int axis = params.axis < 0 ? params.axis + split_dimensions : params.axis;
  int outputs_count = params.num_split;
  TFLITE_DCHECK_LT(axis, split_dimensions);

  int64_t split_size = 0;
  for (int i = 0; i < outputs_count; i++) {
    TFLITE_DCHECK_EQ(output_shapes[i].DimensionsCount(), split_dimensions);
    for (int j = 0; j < split_dimensions; j++) {
      if (j != axis) {
        MatchingDim(output_shapes[i], j, input_shape, j);
      }
    }
    split_size += output_shapes[i].Dims(axis);
  }
  TFLITE_DCHECK_EQ(split_size, input_shape.Dims(axis));
  int64_t outer_size = 1;
  for (int i = 0; i < axis; ++i) {
    outer_size *= input_shape.Dims(i);
  }
  // For all output arrays,
  // FlatSize() = outer_size * Dims(axis) * base_inner_size;
  int64_t base_inner_size = 1;
  for (int i = axis + 1; i < split_dimensions; ++i) {
    base_inner_size *= input_shape.Dims(i);
  }

  const Scalar* input_ptr = input_data;
  for (int k = 0; k < outer_size; k++) {
    for (int i = 0; i < outputs_count; ++i) {
      const int copy_size = output_shapes[i].Dims(axis) * base_inner_size;
      memcpy(output_data[i] + k * copy_size, input_ptr,
             copy_size * sizeof(Scalar));
      input_ptr += copy_size;
    }
  }
}
}

Split::Split(std::vector<std::pair<const Tensor *, int32_t>> &&inputs, std::vector<std::pair<Tensor *, int32_t>> &&outputs)
  : Kernel(std::move(inputs), std::move(outputs))
{
}

void Split::configure(luci::CircleReader *circle_reader, int32_t index)
{
//  assert(axis()->shape().num_elements() == 1);
//  _axis_value = getTensorData<int32_t>(axis())[0];
//  if (_axis_value < 0)
//    _axis_value += input()->shape().num_dims();
//  assert(_axis_value >= 0 && _axis_value < input()->shape().num_dims());
//
//  const int32_t input_size = input()->shape().dim(_axis_value);
//  assert(input_size % _outputs.size() == 0);
//  const int32_t slice_size = input_size / _outputs.size();
//
//  Shape output_shape = input()->shape();
//  output_shape.dim(_axis_value) = slice_size;
//  for (Tensor *output : _outputs)
//  {
//    output->resize(output_shape);
//  }
}

template <typename T> void Split::executeImpl(luci::CircleReader *circle_reader, int32_t index) const
{
  const auto input_tensor_shape = luci::wrap(circle_reader->tensors()[input_ind()]->shape());
  tflite::RuntimeShape input_runtime_shape(input_tensor_shape.size());
  for (int i = 0; i < input_tensor_shape.size(); ++i)
  {
    input_runtime_shape.SetDim(i, input_tensor_shape[i]);
  }

  int _axis_value = getTensorData<int32_t>(axis())[0];

  if (_axis_value < 0)
    _axis_value += input_tensor_shape.size();

  tflite::SplitParams params{};
  params.num_split = _outputs.size();
  params.axis = _axis_value;

  std::vector<tflite::RuntimeShape> output_shapes;
  std::vector<T *> all_data;

  for (uint32_t i = 0; i < _outputs.size(); ++i)
  {
    const auto output_tensor_shape = luci::wrap(circle_reader->tensors()[_outputs[i].second]->shape());
    tflite::RuntimeShape output_runtime_shape(output_tensor_shape.size());
    for (int j = 0; j < output_tensor_shape.size(); ++j)
    {
      output_runtime_shape.SetDim(j, output_tensor_shape[j]);
    }
    output_shapes.push_back(output_runtime_shape);

    all_data.push_back(getTensorData<T>(_outputs.at(i).first));
  }
  Split_impl(params, input_runtime_shape, getTensorData<T>(input()), output_shapes, all_data);
}

void Split::execute(luci::CircleReader *circle_reader, int32_t index) const
{
  auto input_dtype = luci::luci_datatype(circle_reader->tensors()[input_ind()]->type());
  switch (input_dtype)
  {
    case DataType::FLOAT32:
      executeImpl<float>(circle_reader, index);
      break;
    case DataType::U8:
      executeImpl<uint8_t>(circle_reader, index);
      break;
    default:
      throw std::runtime_error("Unsupported type.");
  }
}

} // namespace kernels
} // namespace luci_interpreter
