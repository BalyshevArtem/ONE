/*
 * Copyright (c) 2023 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FuseReshapeMeanMulReshapeDivPatternPass.h"
#include "CircleOptimizerUtils.h"

#include <luci/Profile/CircleNodeOrigin.h>
#include <luci/IR/CircleNodes.h>

namespace
{

std::vector<uint32_t> node_shape(const luci::CircleNode *input)
{
  std::vector<uint32_t> shape;
  uint32_t rank = input->rank();
  for (uint32_t r = 0; r < rank; ++r)
    shape.push_back(input->dim(r).value());

  return shape;
}

luci::CircleConst *create_shape_const(loco::Graph *graph, const std::vector<uint32_t> &new_shape)
{
  // NOTE dim_size can be 0
  uint32_t dim_size = static_cast<uint32_t>(new_shape.size());

  auto shape_const = graph->nodes()->create<luci::CircleConst>();

  // const shape/dtype
  shape_const->dtype(loco::DataType::S32);
  if (dim_size > 0)
  {
    shape_const->rank(1);
    shape_const->dim(0).set(dim_size);
  }
  else
    shape_const->rank(0);
  shape_const->shape_status(luci::ShapeStatus::VALID);

  // constant values
  shape_const->size<loco::DataType::S32>(dim_size);
  for (uint32_t i = 0; i < dim_size; ++i)
    shape_const->at<loco::DataType::S32>(i) = new_shape.at(i);

  return shape_const;
}

} // namespace

namespace luci
{

/**
 * Pass to fuse reshape mean mul reshape div pattern to mean reshape div
 *
 **/
bool FuseReshapeMeanMulReshapeDivPatternPass::run(loco::Graph *g)
{
  bool changed = false;

  auto nodes = loco::active_nodes(loco::output_nodes(g));
  for (auto node : nodes)
  {
    auto div = dynamic_cast<luci::CircleDiv *>(node);
    if (not div)
      continue;

    auto div_const = dynamic_cast<luci::CircleConst *>(div->x());
    if (not div_const)
      continue;

    if (div_const->dtype() != loco::DataType::FLOAT32)
      continue;

    if (div_const->size<loco::DataType::FLOAT32>() != 1)
      continue;

    auto reshape_after_div = dynamic_cast<luci::CircleReshape *>(div->y());
    if (not reshape_after_div)
      continue;

    auto mul = dynamic_cast<luci::CircleMul *>(reshape_after_div->tensor());
    if (not mul)
      continue;

    luci::CircleConst *mul_const = nullptr;
    luci::CircleMean *mean = nullptr;

    mean = dynamic_cast<luci::CircleMean *>(mul->x());
    if (not mean)
    {
      mul_const = dynamic_cast<luci::CircleConst *>(mul->x());
      mean = dynamic_cast<luci::CircleMean *>(mul->y());
    } else
    {
      mul_const = dynamic_cast<luci::CircleConst *>(mul->y());
    }

    if (mul_const == nullptr or mean == nullptr)
      continue;

    if (mul_const->dtype() != loco::DataType::FLOAT32)
      continue;

    if (mul_const->size<loco::DataType::FLOAT32>() != 1)
      continue;

    auto reshape_after_mean = dynamic_cast<luci::CircleReshape *>(mean->input());
    if (not reshape_after_mean)
      continue;

    auto input_node = dynamic_cast<luci::CircleNode *>(reshape_after_mean->tensor());

    if (not input_node)
      continue;

    if (mean->rank() != input_node->rank())
      continue;

    auto div_value = div_const->at<loco::DataType::FLOAT32>(0);
    auto mul_value = mul_const->at<loco::DataType::FLOAT32>(0);

    auto new_value = div_value / mul_value;

    mean->input(input_node);
    reshape_after_div->tensor(mean);
    div_const->at<loco::DataType::FLOAT32>(0) = new_value;

    changed = true;
  }

  return changed;
}

} // namespace luci
