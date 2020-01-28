/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file tensorize.cc
 */

// IR Passes for a generic tensorization code generation.
//
#include <tvm/ir.h>
#include <tvm/expr.h>
#include <tvm/operation.h>
#include <tvm/ir_functor_ext.h>
#include <tvm/expr_operator.h>
#include <tvm/ir_pass.h>
#include <tvm/buffer.h>
#include <tvm/target_info.h>
#include <tvm/build_module.h>
#include <tvm/runtime/device_api.h>
#include <unordered_map>
#include "ir_util.h"
#include "../arithmetic/compute_expr.h"
#include "../runtime/thread_storage_scope.h"

namespace tvm {
namespace ir {

class DfsBinaryOps : public ExprVisitor {
 public:
  struct BinaryInfo {
    int op_code;
    PrimExpr a, b;
    BinaryInfo(const int &op_code, const PrimExpr &a, const PrimExpr &b) : op_code(op_code), a(a), b(b) {}

    bool Homo(const BinaryInfo &other) {
      return op_code == other.op_code && a.dtype() == other.a.dtype() && b.dtype() == other.b.dtype();
    }
  };

  template<typename T>
  void VisitBinaryImpl(const T *op) {
    pre.emplace_back(op->type_index(), op->a, op->b);
    ExprVisitor::VisitExpr(op->a);
    pre.emplace_back(op->type_index(), op->a, op->b);
    ExprVisitor::VisitExpr(op->b);
  }

  using ExprVisitor::VisitExpr;

  std::vector<BinaryInfo> pre;
  std::vector<BinaryInfo> mid;
  // TODO(@were): More arithmetic operations to add
  void VisitExpr_(const AddNode *op) { VisitBinaryImpl(op); }
  void VisitExpr_(const MulNode *op) { VisitBinaryImpl(op); }
};


bool HomoExpr(const PrimExpr &a, const PrimExpr &b) {
  DfsBinaryOps va;
  va.VisitExpr(a);
  DfsBinaryOps vb;
  va.VisitExpr(b);
  if (va.pre.size() != vb.pre.size()) {
    return false;
  }
  auto &apre = va.pre;
  auto &amid = va.mid;
  auto &bpre = vb.pre;
  auto &bmid = vb.mid;

  std::map<PlaceholderOpNode*, PrimExpr> corr;

  for (int i = 0, n = apre.size(); i < n; ++i) {
    if (!apre[i].Homo(bpre[i]) || !amid[i].Homo(bmid[i])) {
      return false;
    }
  }
  return true;
}

class TensorizeAnalyzer : public StmtVisitor {
  ComputeOpNode *op;

  void VisitStmt_(const ProvideNode *provide) override {
    CHECK_EQ(op->body.size(), 1);
    HomoExpr(op->body[0], provide->value);
  }

};


class TensorizeRewriter : public StmtExprMutator {
  int matched{-1};
  const ComputeOpNode *op;

 public:

  Stmt VisitStmt_(const AttrStmtNode *pragma) override {
    if (pragma->attr_key == "tensorize") {
      auto args = pragma->node.as<StrMapNode>();
      op = args->data.at("op").as<ComputeOpNode>();
      matched = -1;
      CHECK_EQ(op->body.size(), 1) << "For now, only one output value is supported";
      op = nullptr;
    }
    return StmtMutator::VisitStmt_(pragma);
  }

};

Stmt RewriteGenericTensorization(Stmt stmt) {
  return TensorizeRewriter()(stmt);
}


}  // namespace ir
}  // namespace tvm
