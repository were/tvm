#include <tvm/runtime/registry.h>
#include <tvm/te/tensor.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/te/operation.h>

#include <unordered_map>
#include <unordered_set>

namespace tvm {
namespace arith {

using namespace tir;

struct OperandInfo {
  int opcode;
  DataType lhs, rhs;
  OperandInfo(int opcode, DataType lhs, DataType rhs) :
    opcode(opcode), lhs(lhs.code(), lhs.bits(), lhs.lanes()),
    rhs(rhs.code(), rhs.bits(), rhs.lanes()) {}
  bool operator!=(const OperandInfo &b) {
    return opcode != b.opcode || lhs != b.lhs || rhs != b.rhs;
  }
};

struct GatherAxis : public StmtExprVisitor {
  std::set<const VarNode *> vs;
  void VisitExpr_(const VarNode *vn) {
    vs.insert(vn);
  }
  void Gather(const PrimExpr &expr) {
    VisitExpr(expr);
  }
};

struct PreMidArithTree : public ExprVisitor {
  int lr{-1};
  CommReducer combiner;
  std::vector<OperandInfo> preorder;
  std::vector<OperandInfo> midorder;
  std::map<const VarNode*, std::vector<std::pair<int, int>>> ctrl;

  template<typename T, int opcode>
  void VisitBinImpl(const T* node) {
    OperandInfo oi(opcode, node->a.dtype(), node->b.dtype());
    lr = 0;
    preorder.push_back(oi);
    VisitExpr(node->a);
    lr = 1;
    // LOG(INFO) << "mid: " << GetRef<PrimExpr>(node);
    midorder.push_back(oi);
    VisitExpr(node->b);
  }

  void VisitExpr_(const AddNode *an) { VisitBinImpl<AddNode, 1>(an); }
  void VisitExpr_(const MulNode *mn) { VisitBinImpl<MulNode, 2>(mn); }
  void VisitExpr_(const SubNode *sn) { VisitBinImpl<SubNode, 3>(sn); }
  void VisitExpr_(const DivNode *dn) { VisitBinImpl<DivNode, 4>(dn); }
  void VisitExpr_(const ReduceNode *cr) {
    combiner = cr->combiner;
    VisitExpr(cr->source[0]);
  }
  void VisitExpr_(const ProducerLoadNode *load) {
    GatherAxis ga;
    for (auto elem : load->indices) {
      ga.Gather(elem);
    }
    for (auto elem : ga.vs) {
      ctrl[elem].emplace_back((int) preorder.size(), lr);
    }
  }

  void ExtractOrder(const PrimExpr &expr) {
    VisitExpr(expr);
  }
};

struct ChooseKFromN {
  int n, k;
  std::vector<int> idx;

  ChooseKFromN(int n_, int k_) : n(n_), k(k_), idx(k) {
    CHECK(n >= k);
    for (int i = 0; i < k; ++i) {
      idx[i] = i;
    }
    --idx.back();
  }

  bool HasNext() {
    for (int i = 0; i < k; ++i)
      if (idx[i] != n - k + i)
        return true;
    return false;
  }

  void Next() {
    CHECK(HasNext());
    ++idx.back();
    int tail = (int) idx.size() - 1, delta = 1;
    while (tail > 0 && idx[tail] > n - delta) {
      idx[tail] = -1;
      ++idx[tail - 1];
      --tail;
      ++delta;
    }
    for (int i = tail + 1; i < k; ++i) {
      idx[i] = idx[i - 1] + 1;
    }
  }

  Array<Integer> ToArray() {
    Array<Integer> res;
    for (int i = 0, n = idx.size(); i < n; ++i)
      res.push_back(idx[i]);
    return res;
  }

  Array<Integer> Reversed() {
    Array<Integer> res;
    for (int i = 0, n = idx.size(); i < n; ++i)
      res.push_back(this->n - idx[i] - 1);
    return res;
  }

};

Array<IterVar> MatchTensorizer(const te::Operation &body, const te::Operation &stencil) {
  Array<IterVar> res;
  auto a = body.as<te::ComputeOpNode>();
  auto b = stencil.as<te::ComputeOpNode>();
  if (a->axis.size() < b->axis.size() || a->reduce_axis.size() < b->reduce_axis.size()) {
    return res;
  }
  CHECK_EQ(a->body.size(), 1);
  CHECK_EQ(b->body.size(), 1);

  /* Check the homo-morphism of the arithmetic tree */
  PreMidArithTree ao, bo;
  ao.ExtractOrder(a->body[0]);
  CHECK_EQ(ao.midorder.size(), ao.preorder.size());
  bo.ExtractOrder(b->body[0]);
  CHECK_EQ(bo.midorder.size(), bo.preorder.size());
  for (int i = 0, n = ao.preorder.size(); i < n; ++i) {
    if (ao.preorder[i] != bo.preorder[i] || ao.midorder[i] != bo.midorder[i]) {
      LOG(INFO) << i << " arith different!";
      return res;
    }
  }

  // TODO(@were): How can we fix the combiner?
  //if (ao.combiner.get() != bo.combiner.get()) {
  //  LOG(INFO) << "different combiner!";
  //  return res;
  //}

  /* Check the homo-morphism of the iteration domain */
  ChooseKFromN axis_enum(a->axis.size(), b->axis.size());
  while (axis_enum.HasNext()) {
    axis_enum.Next();
    ChooseKFromN reduce_enum(a->reduce_axis.size(), b->reduce_axis.size());
    auto scan_idx = axis_enum.Reversed();
    while (reduce_enum.HasNext()) {
      reduce_enum.Next();
      auto reduce_idx = reduce_enum.Reversed();

      auto f = [](const std::vector<std::pair<int, int>> &sub,
                  const std::vector<std::pair<int, int>> &super) {
        if (sub.size() > super.size()) {
          return false;
        }
        for (int i = 0, n = sub.size(); i < n; ++i) {
          bool not_found = true;
          for (int j = 0, m = super.size(); j < m; ++j) {
            if (sub[i] == super[j]) {
              not_found = false;
              break;
            }
          }
          if (not_found) {
            return false;
          }
        }
        return true;
      };
      auto g = [f](Array<Integer> idx, Array<IterVar> axis0, Array<IterVar> axis1,
                   std::map<const VarNode*, std::vector<std::pair<int, int>>> &c0,
                   std::map<const VarNode*, std::vector<std::pair<int, int>>> &c1) {
        for (int i = 0, n = idx.size(); i < n; ++i) {
          auto actual = axis0[idx[i]];
          auto paradigm = axis1[i];
          if (auto ai = actual->dom->extent.as<IntImmNode>()) {
            if (auto pi = paradigm->dom->extent.as<IntImmNode>()) {
              if (ai->value % pi->value) {
                return false;
              }
            }
          }
          if (!f(c0[actual->var.get()], c1[paradigm->var.get()])) {
            return false;
          }
        }
        return true;
      };
      bool ok = g(scan_idx, a->axis, b->axis, ao.ctrl, bo.ctrl) &&
                g(reduce_idx, a->reduce_axis, b->reduce_axis, ao.ctrl, bo.ctrl);
      if (ok) {
        for (int i = 0, n = scan_idx.size(); i < n; ++i) {
          // LOG(INFO) << a->axis[scan_idx[i]] << " -> " << b->axis[i];
          res.push_back(a->axis[scan_idx[i]]);
          res.push_back(b->axis[i]);
        }
        for (int i = 0, n = reduce_idx.size(); i < n; ++i) {
          // LOG(INFO) << a->reduce_axis[reduce_idx[i]] << " -> " << b->reduce_axis[i];
          res.push_back(a->reduce_axis[reduce_idx[i]]);
          res.push_back(b->reduce_axis[i]);
        }
        return res;
      }
    }
  }

  return res;
}

TVM_REGISTER_GLOBAL("arith.MatchTensorizer").set_body_typed(MatchTensorizer);

}  // namespace arith
}  // namespace tvm
