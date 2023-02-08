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

#ifndef LUCI_INTERPRETER_CORE_TENSOR_H
#define LUCI_INTERPRETER_CORE_TENSOR_H

#include "luci_interpreter/core/DataType.h"
#include "luci_interpreter/core/reader/CircleMicroReader.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace luci_interpreter
{

//class Shape
//{
//public:
//  explicit Shape(int rank) : _dims(rank, 0) {}
//
//  Shape(std::initializer_list<int32_t> dims) : _dims(dims.begin(), dims.end()) {}
//
//  int num_dims() const { return _dims.size(); }
//
//  int32_t dim(int i) const
//  {
//    assert(i >= 0 && i < static_cast<int>(_dims.size()));
//    return _dims[i];
//  }
//
//  int32_t &dim(int i)
//  {
//    assert(i >= 0 && i < static_cast<int>(_dims.size()));
//    return _dims[i];
//  }
//
//  int32_t num_elements() const
//  {
//    int32_t result = 1;
//    for (const int32_t dim : _dims)
//    {
//      result *= dim;
//    }
//    return result;
//  }
//
//  bool operator==(const Shape &other) const { return _dims == other._dims; }
//
//  bool operator!=(const Shape &other) const { return !operator==(other); }
//
//private:
//  std::vector<int32_t> _dims;
//};

#ifndef DIS_QUANT
// Tensor affine quantization parameters.
//
// The relationship between real and quantized values:
//   real_value = (quantized_value - zero_point) * scale
//
// In per-tensor case, 'scale' and 'zero_point' are one element each.
// In per-channel case, 'scale' and 'zero_point' are N elements each, where N is the size
// of the quantized dimension.
//
// Note that due to historical and performance reasons, per-tensor quantization uses unsigned
// integer types, while per-channel uses signed types assuming 'zero_point' == 0.
struct AffineQuantization
{
  std::vector<float> scale;
  std::vector<int32_t> zero_point;
  int32_t quantized_dimension;
};
#endif

class Tensor
{
public:
#ifndef DIS_QUANT
  Tensor(DataType element_type, Shape shape, AffineQuantization *quantization);

  float scale() const
  {
    assert(_quantization->scale.size() == 1);
    return _quantization->scale[0];
  }

  int32_t zero_point() const
  {
    assert(_quantization->zero_point.size() == 1);
    return _quantization->zero_point[0];
  }

  const std::vector<float> &scales() const { return _quantization->scale; }

  const std::vector<int32_t> &zero_points() const { return _quantization->zero_point; }

  int32_t quantized_dimension() const { return _quantization->quantized_dimension; }

#else
 // Tensor(const circle::Tensor *raw_tensor);
#endif

  static DataType element_type(const circle::Tensor *_raw_tensor)
  {
    return luci_datatype(_raw_tensor->type());
  }

  static int num_dims(const circle::Tensor *_raw_tensor)
  {
    auto const &const_dims = wrap(_raw_tensor->shape());
    return const_dims.size();
  }

  static int32_t dim(int i, const circle::Tensor *_raw_tensor)
  {
    assert(i >= 0);
    auto const &const_dims = wrap(_raw_tensor->shape());
    assert(i < const_dims.size());

    return const_dims[i];
  }

  static int32_t num_elements(const circle::Tensor *_raw_tensor)
  {
    int32_t result = 1;
    auto const &const_dims = wrap(_raw_tensor->shape());
    for (const int32_t dim : const_dims)
    {
      result *= dim;
    }
    return result;
  }

  //const Shape &shape() const { return _shape; }


//  template <typename T> const T *data() const
//  {
//    static_assert(std::is_same<uint8_t, char>::value or
//                  std::is_same<uint8_t, unsigned char>::value);
//    return reinterpret_cast<const T *>(_data);
//  }
//
//  template <typename T> T *data()
//  {
//    static_assert(std::is_same<uint8_t, char>::value or
//                  std::is_same<uint8_t, unsigned char>::value);
//    return reinterpret_cast<T *>(_data);
//  }
//
  //static void readData(const circle::Tensor *raw_tensor, void *data_ptr, size_t data_size);

  //static void writeData(const circle::Tensor *raw_tensor, const void *data_ptr, size_t data_size);
//
//  void writeDataWithoutCopy(void *data_ptr)
//  {
//    assert(data_ptr != nullptr);
//    _data = static_cast<uint8_t *>(data_ptr);
//  }

  //void resize(const Shape &new_shape);

//  void set_data_buffer(uint8_t *buffer)
//  {
//    _data = buffer;
//  }

//  bool is_allocatable() const { return _is_allocatable; }

//  void set_allocatable(bool value) { _is_allocatable = value; }

  //bool is_data_allocated() const { return _data != nullptr; }

  //uint32_t get_offset() const { return _offset; }

  //void set_offset(uint32_t offset) { _offset = offset; }

private:

#ifndef DIS_QUANT
  AffineQuantization *_quantization;
#endif
  //uint8_t *_data;
  // Memory manager is called for tensor only if it is "allocatable".
  // Kernel configuration could disable allocation of some tensors if they are not needed for
  // particular operation.
 // bool _is_allocatable = true;
  //uint32_t _offset = 0;
 // const circle::Tensor *_raw_tensor;
};

} // namespace luci_interpreter

#endif // LUCI_INTERPRETER_CORE_TENSOR_H
