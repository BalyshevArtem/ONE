/*
 * Copyright (c) 2021 Samsung Electronics Co., Ltd. All Rights Reserved
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

#include "luci/Pass/FoldAddV2Pass.h"

#include <luci/IR/CircleNodes.h>

namespace
{

bool same_shape(const luci::CircleConst *x, const luci::CircleConst *y)
{
  if (x->rank() != y->rank())
    return false;

  for (uint32_t i = 0; i < x->rank(); i++)
  {
    if (!(x->dim(i) == y->dim(i)))
      return false;
  }

  return true;
}

/**
 * Fold AddV2 to const if both inputs are const
 **/
template <loco::DataType T> bool fold_add_v2(luci::CircleCustom *add_v2)
{
  // This should hold for AddV2
  if (add_v2->numInputs() != 2)
    return false;

  // Check first input is const
  auto x = dynamic_cast<luci::CircleConst *>(add_v2->inputs(0));
  if (not x)
    return false;

  // Check second input is const
  auto y = dynamic_cast<luci::CircleConst *>(add_v2->inputs(1));
  if (not y)
    return false;

  if (x->dtype() != y->dtype())
    return false;

  if (!same_shape(x, y))
    return false;

  auto name_x = x->name();
  auto name_y = y->name();
  assert(name_x.length() > 0);
  assert(name_y.length() > 0);
  auto constant = add_v2->graph()->nodes()->create<luci::CircleConst>();
  constant->dtype(x->dtype());
  constant->rank(x->rank());
  for (uint32_t i = 0; i < x->rank(); i++)
    constant->dim(i).set(x->dim(i).value());

  const auto size = x->size<T>();
  constant->size<T>(size);
  for (uint32_t i = 0; i < size; i++)
    constant->at<T>(i) = x->at<T>(i) + y->at<T>(i);

  constant->shape_status(luci::ShapeStatus::VALID);
  constant->name(name_x + ";" + name_y);

  for (auto succ : loco::succs(add_v2))
  {
    auto custom_out = loco::must_cast<luci::CircleCustomOut *>(succ);
    loco::replace(custom_out).with(constant);
  }

  return true;
}

} // namespace

namespace luci
{

/**
 * Constant Folding for AddV2 Op
 **/
//bool FoldAddV2Pass::run(loco::Graph *g)
//{
//  bool changed = false;
//  for (auto node : loco::active_nodes(loco::output_nodes(g)))
//  {
//    if (auto mul = dynamic_cast<luci::CircleInput *>(node))
//    {
//      if (mul->rank() != 3)
//        continue;
//
//      auto l = mul->dim(1);
//      auto r = mul->dim(2);
//
//      mul->rank(2);
//      mul->dim(0) = l;
//      mul->dim(1) = r;
//
//      auto output = dynamic_cast<luci::CircleOutput *>(output_nodes(node->graph())[0]);
//      auto l_o = output->dim(1);
//      auto r_o = output->dim(2);
//
//      output->rank(2);
//      output->dim(0) = l_o;
//      output->dim(1) = r_o;
//
//      g->outputs()->at(0)->shape({l_o, r_o});
//
//      changed = true;
//    }
//    else if (auto ss = dynamic_cast<luci::CircleStridedSlice *>(node))
//    {
//      auto begin = dynamic_cast<luci::CircleConst *>(ss->begin());
//      auto end = dynamic_cast<luci::CircleConst *>(ss->end());
//      auto strides = dynamic_cast<luci::CircleConst *>(ss->strides());
//
//      if (begin->dim(0) == 3)
//      {
//        auto l_b = begin->at<loco::DataType::S32>(1);
//        auto r_b = begin->at<loco::DataType::S32>(2);
//        begin->size<loco::DataType::S32>(2);
//        begin->at<loco::DataType::S32>(0) = l_b;
//        begin->at<loco::DataType::S32>(1) = r_b;
//        begin->dim(0) = 2;
//        changed = true;
//      }
//      if (end->dim(0) == 3)
//      {
//        auto l_e = end->at<loco::DataType::S32>(1);
//        auto r_e = end->at<loco::DataType::S32>(2);
//        end->size<loco::DataType::S32>(2);
//        end->at<loco::DataType::S32>(0) = l_e;
//        end->at<loco::DataType::S32>(1) = r_e;
//        end->dim(0) = 2;
//        changed = true;
//      }
//
//      if (strides->dim(0) == 3)
//      {
//        auto l_s = strides->at<loco::DataType::S32>(1);
//        auto r_s = strides->at<loco::DataType::S32>(2);
//
//        strides->size<loco::DataType::S32>(2);
//        strides->at<loco::DataType::S32>(0) = l_s;
//        strides->at<loco::DataType::S32>(1) = r_s;
//        strides->dim(0) = 2;
//        changed = true;
//      }
//    }
//    else if (auto tr = dynamic_cast<luci::CircleTranspose *>(node))
//    {
//      auto perm = dynamic_cast<luci::CircleConst *>(tr->perm());
//      if (perm->dim(0).value() != 3)
//        continue;
//
//      perm->size<loco::DataType::S32>(2);
//      perm->dim(0) = 2;
//      perm->at<loco::DataType::S32>(0) = 1;
//      perm->at<loco::DataType::S32>(1) = 0;
//      changed = true;
//    }
//    else if (auto add = dynamic_cast<luci::CircleAdd *>(node))
//    {
//      auto left_node = dynamic_cast<luci::CircleConst *>(add->y());
//      if (left_node == nullptr)
//        continue;
//      auto rank = left_node->rank();
//      if (left_node->rank() != 3)
//        continue;
//
//      auto l = left_node->dim(1);
//      auto r = left_node->dim(2);
//      left_node->rank(2);
//      left_node->dim(0) = l;
//      left_node->dim(1) = r;
//
//      changed = true;
//    }
//    else if (auto mul_r = dynamic_cast<luci::CircleMul *>(node))
//    {
//      auto left_node = dynamic_cast<luci::CircleConst *>(mul_r->y());
//      if (left_node == nullptr)
//        continue;
//      auto rank = left_node->rank();
//      if (left_node->rank() != 3)
//        continue;
//
//      auto l = left_node->dim(1);
//      auto r = left_node->dim(2);
//      left_node->rank(2);
//      left_node->dim(0) = l;
//      left_node->dim(1) = r;
//
//      changed = true;
//    }
//  }
//
//  return changed;
//}

// Create npu part
#if 0
bool FoldAddV2Pass::run(loco::Graph *g)
{
  bool changed = false;

  auto f_output = dynamic_cast<luci::CircleOutput *>(output_nodes(g).at(0));

  if (dynamic_cast<luci::CircleConcatenation *>(f_output->from()) != nullptr)
    return false;

  auto concat_node = f_output->graph()->nodes()->create<luci::CircleConcatenation>(g->outputs()->size());
  concat_node->axis(-1);
  concat_node->rank(f_output->rank());
  concat_node->shape_status(ShapeStatus::VALID);
  concat_node->dtype(loco::DataType::FLOAT32);
  concat_node->name("df");
  concat_node->fusedActivationFunction(FusedActFunc::NONE);
  int i = 0;
  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto out = dynamic_cast<luci::CircleOutput *>(node))
    {
      auto mul = dynamic_cast<luci::CircleBatchMatMul *>(out->from());
      if (mul == nullptr)
        continue;
      concat_node->values(i, mul);
      i++;
    }
  }

  g->outputs()->at(0)->shape({f_output->dim(0).value(), f_output->dim(1).value() * i});

  f_output->dim(1) = f_output->dim(1).value() * i;
  f_output->from(concat_node);
//
//  for (int j = 1; j < g->outputs()->size(); ++j)
//  {
//    auto c = g->outputs()->_pool;
//
//
//    auto cur = dynamic_cast<luci::CircleOutput *>(output_nodes(g).at(j));
//    cur->from(nullptr);
//  }

  auto &c = g->outputs()->_pool;

  c.erase(c.begin() + 1, c.end());

  return changed;
}
#endif

// Create onert part
const int NUM = 7;
bool FoldAddV2Pass::run(loco::Graph *g)
{
  bool changed = false;

  for (auto node : loco::active_nodes(loco::output_nodes(g)))
  {
    if (auto concate = dynamic_cast<luci::CircleConcatenation *>(node))
    {
      if (concate->numValues() <= 2 or concate->numValues() == 7)
        continue;

      auto concat_node = concate->graph()->nodes()->create<luci::CircleConcatenation>(NUM + 1);
      concat_node->axis(-1);
      concat_node->rank(concate->rank());
      concat_node->shape_status(ShapeStatus::VALID);
      concat_node->dtype(loco::DataType::FLOAT32);
      concat_node->name(concate->name());
      concat_node->fusedActivationFunction(FusedActFunc::NONE);

      for (int i = 0; i < NUM; ++i)
      {
        concat_node->values(i, concate->values(i + 11));
      }

      auto input_new = concate->graph()->nodes()->create<luci::CircleInput>();
      auto input = g->inputs()->create();
      concat_node->values(NUM, input_new);
      input_new->index(1);
      input_new->name("sdf");
      input_new->dtype(loco::DataType::FLOAT32);
      input_new->shape_status(ShapeStatus::VALID);
      input_new->shape({8, 880});

      link(input, input_new);
      input->name("sdf");
      input->shape({8, 880});
      input->dtype(loco::DataType::FLOAT32);

      replace(concate).with(concat_node);
    }
  }


  auto &c = g->outputs()->_pool;

  c.erase(c.begin() + 1, c.end());

  return changed;
}

} // namespace luci
