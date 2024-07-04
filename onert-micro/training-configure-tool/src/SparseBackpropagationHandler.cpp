/*
 * Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "SparseBackpropagationHandler.h"

#include <fstream>
#include <vector>
#include <cstring>

#define HARDCODE_SIZE 24

namespace
{

constexpr uint16_t MAGIC_NUMBER = 29;
constexpr uint8_t SCHEMA_VERSION = 1;

std::vector<uint16_t> hardcode_layers_ops = {11, 7, 3};

void createTmpFile(std::vector<char> &buffer)
{
  // Resize to calculated size
  auto cur_size = 8 + hardcode_layers_ops.size() * 2;
  buffer.resize(HARDCODE_SIZE);

  // Point to start of the buffer
  char *cur_ptr = buffer.data();

  // Write MAGIC_NUMBER
  std::memcpy(cur_ptr, &MAGIC_NUMBER, sizeof(MAGIC_NUMBER));
  cur_ptr += 2;

  // Write SCHEMA_VERSION
  std::memcpy(cur_ptr, &SCHEMA_VERSION, sizeof(SCHEMA_VERSION));
  cur_ptr += 1;

  // Miss RESERVED field
  cur_ptr += 1;

  // Write number of layers
  auto layers_num = static_cast<int32_t>(hardcode_layers_ops.size());
  std::memcpy(cur_ptr, &layers_num, sizeof(layers_num));
  cur_ptr += 4;

  // Write trainable layers positions
  for (uint16_t cur_layer_pos : hardcode_layers_ops)
  {
    std::memcpy(cur_ptr, &cur_layer_pos, sizeof(cur_layer_pos));
    cur_ptr += 2;
  }
}

} // namespace

onert_micro::OMStatus training_configure_tool::createResultFile(const char *save_path)
{
  std::vector<char> tmp_buffer;

  createTmpFile(tmp_buffer);

  // Open or create file
  // Note: if the file existed, it will be overwritten
  std::ofstream out_file(save_path, std::ios::binary | std::ios::trunc);
  if (not out_file.is_open())
    return onert_micro::UnknownError;

  // Write data
  out_file.write(tmp_buffer.data(), tmp_buffer.size());

  // Close file
  out_file.close();

  return onert_micro::Ok;
}

onert_micro::OMStatus training_configure_tool::createResultData(std::vector<char> &result_buffer)
{
  createTmpFile(result_buffer);

  return onert_micro::Ok;
}
