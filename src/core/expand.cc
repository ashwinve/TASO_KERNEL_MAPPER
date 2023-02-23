#include "taso/ops.h"
using namespace taso;

/* Copyright 2019 Stanford
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Enlarge the third and forth dimension of _w1 to the same size as _w2
TensorHandle Graph::expand(const TensorHandle _input,
                            const std::vector<int> &_shape)
{
    std::vector<int> out_shape = _shape;
    Op op = model->get_or_create_expand(*_input, out_shape);
    
    add_edge(_input->op, op, _input->idx, 0);
    
    TensorHandle t = new Tensor(op.ptr->outputs[0]);
    t->op = op;
    return t;
}

Op Model::get_or_create_expand(Tensor _input, const std::vector<int>& _shape)
{
  ExpandKey key(_input, _shape);
  Expand* expandOp;
  if (expand.find(key) != expand.end()) {
    expandOp = expand[key];
  } else {
    expandOp = new Expand(this, _input, _shape);
    measure_expand_cost(expandOp);
    expand[key] = expandOp;
  }
  Op ret;
  ret.guid = global_unique_id ++;
  ret.ptr = expandOp;
  return ret;
}

Expand::Expand(Model* _model, Tensor _input, const std::vector<int>& _shape)
: OpBase(_input, _model, OP_EXPAND)
{
  int size = 1;
  // set dims and strides
  numOutputs = 1;
  outputs[0].numDim = _shape.size();
  for (int i = _shape.size() - 1; i >= 0; i--) {
    outputs[0].dim[i] = _shape[i];
    outputs[0].stride[i] = size;
    size *= _shape[i];
    outputs[0].split[i] = SplitInfo::NO_SPLIT;
  }

  outputs[0].idx = 0;
}

Expand::~Expand(void)
{}

bool Expand::get_int_parameter(PMParameter para, int* value)
{
  switch (para) {
    default:
      return OpBase::get_int_parameter(para, value);
  }
}

void Expand::collect_costs(float& exe_time, float& flops,
                            float& mem_acc, int& num_kernels)
{
  exe_time += runtime;
}

ExpandKey::ExpandKey(Tensor _input, const std::vector<int>& shape)
{
  int idx = 0;
  keys[idx++] = shape.size();
  for (size_t i = 0; i < shape.size(); i++)
    keys[idx++] = shape[i];
  _input.serialize(keys, idx);
  while (idx < KEY_LENGTH)
    keys[idx++] = 0;
  assert(idx == KEY_LENGTH);
}

