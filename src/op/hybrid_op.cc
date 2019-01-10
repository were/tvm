/*!
 *  Copyright (c) 2019 by Contributors
 * \brief Hybrid computation rule.
 * \file hybrid_op.cc
 */
#include <tvm/operation.h>
#include <tvm/arithmetic.h>
#include <tvm/ir.h>
#include <tvm/ir_mutator.h>
#include <tvm/ir_operator.h>
#include <tvm/ir_pass.h>
#include <unordered_set>

#include "op_util.h"
#include "hybrid_op.h"

namespace tvm {
using namespace ir;
// HybridOpNode
TVM_STATIC_IR_FUNCTOR(IRPrinter, vtable)
.set_dispatch<HybridOpNode>([](const HybridOpNode *op, IRPrinter *p) {
    p->stream << "hybrid(" << op->name << ", " << op << ")";
  });

TVM_REGISTER_NODE_TYPE(HybridOpNode);

int HybridOpNode::num_outputs() const {
  return static_cast<int>(outputs.size());
}

Array<IterVar> HybridOpNode::root_iter_vars() const {
  return this->axis;
}

Type HybridOpNode::output_dtype(size_t i) const {
  return outputs[i]->dtype;
}

Array<Expr> HybridOpNode::output_shape(size_t i) const {
  return outputs[i]->shape;
}


Operation HybridOpNode::make(std::string name,
                             std::string tag,
                             Map<std::string, NodeRef> attrs,
                             Array<Tensor> inputs,
                             Array<Tensor> outputs,
                             Stmt body) {
  if (!attrs.defined()) {
    attrs = Map<std::string, NodeRef>();
  }
  auto n = make_node<HybridOpNode>();
  n->name = std::move(name);
  n->tag = std::move(tag);
  n->attrs = std::move(attrs);
  n->inputs = std::move(inputs);
  n->outputs = std::move(outputs);
  n->axis = op::GatherLoopVars(body);
  n->body = std::move(body);
  Operation res = Operation(n);
  return res;
}

Array<Tensor> HybridOpNode::InputTensors() const {
  return inputs;
}

Operation HybridOpNode::ReplaceInputs(
    const Operation& self,
    const std::unordered_map<Tensor, Tensor>& rmap) const {
  CHECK_EQ(self.operator->(), this);
  auto n = make_node<HybridOpNode>(*this);
  n->body = op::ReplaceTensor(this->body, rmap);
  for (size_t i = 0; i < n->inputs.size(); ++i) {
    Tensor t = n->inputs[i];
    if (rmap.count(t)) {
      n->inputs.Set(i, rmap.at(t));
    }
  }

  if (body.same_as(n->body) &&
      inputs.same_as(n->inputs)) {
    return self;
  } else {
    return Operation(n);
  }
}

void HybridOpNode::PropBoundToInputs(
    const Operation& self,
    const std::unordered_map<const Variable*, IntSet>& dom_map,
    std::unordered_map<Tensor, TensorDom>* out_dom_map) const {
  for (Tensor t : this->inputs) {
    auto it = out_dom_map->find(t);
    if (it == out_dom_map->end()) continue;
    TensorDom& dom = it->second;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      dom.data[i].emplace_back(IntSet::range(
          Range::make_by_min_extent(
              make_const(t->shape[i].type(), 0), t->shape[i])));
    }
  }
}

void HybridOpNode::GatherBound(
    const Operation& self,
    const std::unordered_map<Tensor, TensorDom>& tensor_dom,
    std::unordered_map<IterVar, Range>* out_dom_map) const {
  for (auto iter_var : axis) {
    CHECK(!out_dom_map->count(iter_var));
    out_dom_map->operator[](iter_var) = iter_var->dom;
  }
}

Stmt HybridOpNode::BuildRealize(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& realize_map,
    const Stmt& body) const {
  CHECK_EQ(stage->op.get(), this);
  Stmt realize_body = body;
  for (int k = 0; k < num_outputs(); ++k) {
    Tensor t = stage->op.output(k);
    HalideIR::Internal::Region bounds;
    for (size_t i = 0; i < t->shape.size(); ++i) {
      bounds.push_back(
          Range::make_by_min_extent(
              make_const(t->shape[i].type(), 0), t->shape[i]));
    }
    realize_body = ir::Realize::make(
        t->op, t->value_index, t->dtype,
        bounds, const_true(), realize_body);
  }
  return realize_body;
}

Stmt HybridOpNode::BuildProvide(
    const Stage& stage,
    const std::unordered_map<IterVar, Range>& dom_map,
    bool debug_keep_trivial_loop) const {
  CHECK_EQ(stage->op.operator->(), this);
  Stmt ret = AttrStmt::make(make_zero(Int(32)), attr::extern_scope, 0, this->body);
  auto f_push_bind = [&ret](Buffer buffer, Tensor tensor) {
    Array<NodeRef> bind_spec;
    Array<Expr> tuple;
    bind_spec.push_back(buffer);
    bind_spec.push_back(tensor);
    for (size_t k = 0; k < buffer->shape.size(); ++k) {
      tuple.push_back(make_const(buffer->shape[k].type(), 0));
      tuple.push_back(buffer->shape[k]);
    }
    ret = AttrStmt::make(
        bind_spec, attr::buffer_bind_scope,
        Call::make(Handle(), intrinsic::tvm_tuple, tuple, Call::Intrinsic), ret);
  };
  for (int i = static_cast<int>(outputs.size()) - 1; i >= 0; --i) {
    Buffer buffer = decl_buffer(
      outputs[i]->shape,
      outputs[i]->dtype);
    f_push_bind(buffer, stage->op.output(i));
  }
  for (int i = static_cast<int>(inputs.size()) - 1; i >= 0; --i) {
    Buffer buffer = decl_buffer(
      inputs[i]->shape,
      inputs[i]->dtype);
    f_push_bind(buffer, inputs[i]);
  }

  std::unordered_map<Tensor, Tensor> rmap;
  for (int i = 0; i < this->num_outputs(); ++i) {
    rmap[outputs[i]] = stage->op.output(i);
  }
  auto n = make_node<HybridOpNode>(*this);
  /*
   * These two lines of codes replace tensors' reads & writes.
   * This is the simplest way I (@were) can come up with to glue
   * hybrid scripts to the structure of TVM op.
   * NAMING CONFLICT: In hybrid script all the tensors have their own 
   * names specified by the users. However, In TVM op, all the output
   * tensors' names are the same as the op's name. I cannot change the
   * name to the op's name in the function body after the op node is
   * formed, because:
   *   1. Output tensors all point to the corresponding op node. 
   *   2. Once OpNode is wrapped up by an Operation node, it can
   *      no longer be changed.
   * This is a chiken-egg paradox. It is impossible to put the output
   * tensors into the function body without forming the op node. The
   * function body is immutable after the node is formed.
   *
   * Finally, I decided to resolve this issue "lazily". During the
   * pipeline of compilation, these tensors will be replaced when
   * forming the function body and passing to next stage of compilation.
   * */
  ret = op::ReplaceTensor(ret, rmap);
  ret = op::ReplaceProvideTensor(ret, rmap);

  ret = op::ApplySchedule(stage, dom_map, ret);
  return ret;
}

namespace op {

Stmt ApplySplits(const Stage &stage,
                 const std::unordered_map<IterVar, Range>& dom_map, Stmt stmt) {
  class LoopSpliter : public IRMutator {
    Expr factor;
    IterVar parent, inner, outer;
   public:
    LoopSpliter(const SplitNode *split,
                const std::unordered_map<IterVar, Range>& dom_map) : 
      factor(split->factor) {

      auto &parent_ = split->parent;
      if (parent_->dom.defined()) {
        CHECK(is_const_int(parent_->dom->min, 0));
        parent= parent_;
      } else {
        CHECK(dom_map.count(parent_));
        auto &dom = dom_map.find(parent_)->second;
        CHECK(is_const_int(dom->min, 0));
        parent = IterVarNode::make(dom, parent_->var, parent_->iter_type);
      }

      auto &inner_ = split->inner;
      CHECK(dom_map.count(inner_));
      auto &inner_dom = dom_map.find(inner_)->second;
      CHECK(is_const_int(inner_dom->min, 0));

      auto &outer_ = split->outer;
      CHECK(dom_map.count(outer_));
      auto &outer_dom = dom_map.find(outer_)->second;
      CHECK(is_const_int(outer_dom->min, 0));

      inner = IterVarNode::make(inner_dom, inner_->var, inner_->iter_type);
      outer = IterVarNode::make(outer_dom, outer_->var, outer_->iter_type);
    }

    Stmt Mutate_(const For *op, const Stmt &stmt) {
      if (op->loop_var.get() == parent->var.get()) {
        std::unordered_map<const Variable *, Expr> rmap;
        rmap[op->loop_var.get()] = inner + outer * factor;
        Stmt ret = ir::Substitute(op->body, rmap);
        Expr cond = likely(outer * factor < (parent->dom->extent - inner));
        ret = IfThenElse::make(cond, ret);
        ret = For::make(inner->var, Expr(0), inner->dom->extent,
                        IterVarTypeToForType(inner->iter_type), op->device_api, ret);
        ret = For::make(outer->var, Expr(0), outer->dom->extent,
                        IterVarTypeToForType(outer->iter_type), op->device_api, ret);
        return ret;
      }
      return IRMutator::Mutate_(op, stmt);
    }
  };

  bool changed = true;
  while (changed) {
    changed = false;
    for (auto &rel : stage->relations) {
      if (const SplitNode* split = rel.as<SplitNode>()) {
        bool not_splited = false;
        PostOrderVisit(stmt, [&not_splited, &split](const NodeRef &node) {
          if (const Variable *var = node.as<Variable>()) {
            if (var == split->parent->var.get())
              not_splited = true;
          }
        });
        if (not_splited) {
          stmt = LoopSpliter(split, dom_map).Mutate(stmt);
          changed = true;
        }
      }
    }
  }

  return stmt;
}

Stmt ApplyLoopAnnotations(const Stage &stage, Stmt stmt) {
  class LoopAnnotator : public IRMutator {
    const Variable *var;
    ForType for_type;
   public:
    LoopAnnotator(const Variable *var_, ForType for_type_) : var(var_), for_type(for_type_) {}
    
    Stmt Mutate_(const For *op, const Stmt &stmt) {
      if (op->loop_var.get() == var) {
        CHECK(for_type != op->for_type);
        return For::make(op->loop_var, op->min, op->extent,
                         for_type, op->device_api, op->body);
      }
      return IRMutator::Mutate_(op, stmt);
    }
  };

  for (auto &iter_var : stage->leaf_iter_vars) {
    bool equal = false;
    int found = 0;

    const Variable *var = iter_var->var.get();
    ForType expected = IterVarTypeToForType(iter_var->iter_type);
    if (stage->iter_var_attrs.count(iter_var)) {
      expected = IterVarTypeToForType(stage->iter_var_attrs[iter_var]->iter_type);
    }

    PostOrderVisit(stmt, [&found, &var, &expected, &equal](const NodeRef &node) {
      if (const For *op = node.as<For>()) {
        if (op->loop_var.get() == var) {
          ++found;
          equal = expected == op->for_type;
        }
      }
    });

    CHECK_EQ(found, 1) << " iter var should be found exactly once!";
    if (!equal) {
      stmt = LoopAnnotator(var, expected).Mutate(stmt);
    }
  }
  return stmt;
}

Stmt ApplyLoopOrder(const Stage &stage, Stmt stmt) {
  return stmt;
}

Stmt ApplySchedule(const Stage &stage, const
                   std::unordered_map<IterVar, Range>& dom_map, Stmt stmt) {
  stmt = ApplySplits(stage, dom_map, stmt);
  stmt = ApplyLoopAnnotations(stage, stmt);
  stmt = ApplyLoopOrder(stage, stmt);
  return stmt;
}

std::vector<IterVar> GatherLoopVars(Stmt stmt) {
  std::vector<IterVar> res_;
  PostOrderVisit(stmt, [&res_](const NodeRef &node) {
    if (const For *op = node.as<For>()) {
      Var loop_var(op->loop_var);
      Range dom = Range::make_by_min_extent(op->min, op->extent);
      res_.push_back(IterVarNode::make(dom, loop_var, ForTypeToIterVarType(op->for_type)));
    }
  });
  return res_;
}

// replacer to replace tensors' usage in Provide
class ProviderReplacer : public ir::IRMutator {
 public:
  explicit ProviderReplacer(const std::unordered_map<Tensor, Tensor>& vmap)
      : vmap_(vmap) {}

  Stmt Mutate_(const ir::Provide* op, const Stmt& s) {
    Tensor t = Operation(op->func.node_).output(op->value_index);
    auto it = vmap_.find(t);
    if (it != vmap_.end()) {
      Stmt ret = ir::Provide::make(
        it->second->op, it->second->value_index, op->value, op->args);
      found = true;
      return IRMutator::Mutate_(ret.as<ir::Provide>(), ret);
    }
    return IRMutator::Mutate_(op, s);
  }

  // whether it is found.
  bool found{false};

 private:
  const std::unordered_map<Tensor, Tensor>& vmap_;
};

Stmt ReplaceProvideTensor(Stmt stmt,
                   const std::unordered_map<Tensor, Tensor>& replace) {
  ProviderReplacer repl(replace);
  Stmt ret = repl.Mutate(stmt);
  return repl.found ? ret : stmt;
}
} // namespace op
}  // namespace tvm
