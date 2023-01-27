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

#include "luci_interpreter/core/Tensor.h"

#include <cstring>

namespace luci_interpreter
{
#ifndef DIS_QUANT
Tensor::Tensor(DataType element_type, Shape shape, AffineQuantization *quantization)
  : _element_type(element_type), _shape(std::move(shape)), _quantization(quantization),
    _data_allocated(false), _data(nullptr)
{
}
#else

Tensor::Tensor(const circle::Tensor *raw_tensor)
  : _raw_tensor(raw_tensor), _is_allocatable(true), _data(nullptr)
{
  // Do nothing
}

//Tensor::Tensor(DataType element_type, Shape shape)
//  : _element_type(element_type), _shape(std::move(shape)),
//    _data_allocated(false), _data(nullptr)
//{
//}
#endif

void Tensor::readData(void *data_ptr, size_t data_size) const
{
  const size_t element_size = getDataTypeSize(element_type());
  const int32_t num_elements_value = num_elements();
  if (data_size != num_elements_value * element_size)
  {
    assert(false && "Invalid data size.");
  }
  assert(data_ptr != nullptr);
  std::memcpy(data_ptr, data<void>(), data_size);
}

void Tensor::writeData(const void *data_ptr, size_t data_size)
{
  const size_t element_size = getDataTypeSize(element_type());
  const int32_t num_elements_value = num_elements();
  if (data_size != num_elements_value * element_size)
  {
    assert(false && "Invalid data size.");
  }
  assert(data_ptr != nullptr);
  std::memcpy(data<void>(), data_ptr, data_size);
}

// TODO: enable this
//void Tensor::resize(const Shape &new_shape) { _shape = new_shape; }

} // namespace luci_interpreter
