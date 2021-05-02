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
 * \file codegen_riscv64.cc
 * \brief For my DSA-extended riscv64 target.
 */
#ifdef TVM_LLVM_VERSION

#include <tvm/runtime/registry.h>

#include "codegen_cpu.h"
#include "llvm/MC/MCSubtargetInfo.h"

namespace tvm {
namespace codegen {

class CodeGenRISCV64 final : public CodeGenCPU {
 public:
  llvm::Function *dsa_config_start;
  llvm::Function *dsa_config_end;

  void Init(const std::string& module_name, llvm::TargetMachine* tm, llvm::LLVMContext* ctx,
            bool system_lib, bool dynamic_lookup, bool target_c_runtime) override {
    CodeGenCPU::Init(module_name, tm, ctx, system_lib, dynamic_lookup, target_c_runtime);
    dsa_config_start =
      llvm::Intrinsic::getDeclaration(module_.get(), llvm::Intrinsic::ss_config_start);
    dsa_config_end =
      llvm::Intrinsic::getDeclaration(module_.get(), llvm::Intrinsic::ss_config_end);
  }

  void VisitStmt_(const AttrStmtNode* op) override {
    auto f = [](ObjectRef obj, int64_t dft) {
      if (obj.defined()) {
        auto imn = obj.as<IntImmNode>();
        return imn ? imn->value : dft;
      }
      return dft;
    };
    llvm::Value *config_start_call = nullptr;
    if (op->attr_key == "pragma_dsa.offload") {
      int unroll = f(op->value, 1);
      llvm::Metadata *values[] = {
        llvm::MDString::get(*ctx_, "llvm.loop.ss.dedicated"),
        llvm::ConstantAsMetadata::get(builder_->getInt32(unroll)),
      };
      lmd_.push_back(llvm::MDNode::get(*ctx_, values));
    } else if (op->attr_key == "pragma_dsa.stream") {
      int block = f(op->value, 1);
      llvm::Metadata *values[] = {
        llvm::MDString::get(*ctx_, "llvm.loop.ss.stream"),
        llvm::ConstantAsMetadata::get(builder_->getInt32(block)),
      };
      lmd_.push_back(llvm::MDNode::get(*ctx_, values));
    } else if (op->attr_key == "pragma_dsa.config") {
      config_start_call = builder_->CreateCall(dsa_config_start, {});
    }
    CodeGenCPU::VisitStmt_(op);
    if (op->attr_key == "pragma_dsa.config") {
      ICHECK(config_start_call);
      builder_->CreateCall(dsa_config_end, {config_start_call});
    }
  }

};


TVM_REGISTER_GLOBAL("tvm.codegen.llvm.target_riscv64")
    .set_body([](const TVMArgs& targs, TVMRetValue* rv) {
      CodeGenLLVM* cg = new CodeGenRISCV64();
      *rv = static_cast<void*>(cg);
    });

}  // namespace codegen
}  // namespace tvm
#endif  // TVM_LLVM_VERSION
