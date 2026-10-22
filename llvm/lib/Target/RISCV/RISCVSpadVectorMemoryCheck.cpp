//===- RISCVSpadVectorMemoryCheck.cpp - Vector memory reaches SPAD only ---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// On a spad-vector-memory machine a vector memory instruction adds a per-lane
// bank offset to a lane-0 address, so it may address only the scratchpad
// (address space RISCVAS::SPAD) or a spill slot of the function's own frame,
// and never through an indexed or fault-only-first addressing mode. This pass
// refuses, with an error diagnostic, every vector memory instruction that
// codegen produced outside those bounds.
//
// It runs after register allocation, so spills and folded reloads are visible,
// and before frame lowering, so stack accesses still name their frame index.
//
//===----------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVRegisterInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/LLVMContext.h"

using namespace llvm;

#define DEBUG_TYPE "riscv-spad-vector-memory-check"
#define RISCV_SPAD_VECTOR_MEMORY_CHECK_NAME "RISC-V spad vector memory check"

namespace {
class RISCVSpadVectorMemoryCheck : public MachineFunctionPass {
public:
  static char ID;

  RISCVSpadVectorMemoryCheck() : MachineFunctionPass(ID) {}
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesAll();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override {
    return RISCV_SPAD_VECTOR_MEMORY_CHECK_NAME;
  }
};
} // end anonymous namespace

char RISCVSpadVectorMemoryCheck::ID = 0;
INITIALIZE_PASS(RISCVSpadVectorMemoryCheck, DEBUG_TYPE,
                RISCV_SPAD_VECTOR_MEMORY_CHECK_NAME, false, false)

FunctionPass *llvm::createRISCVSpadVectorMemoryCheckPass() {
  return new RISCVSpadVectorMemoryCheck();
}

// A memory instruction is a vector memory instruction when its descriptor
// gives one of its operands an RVV register class: unit-stride, strided,
// indexed, segment, mask, whole-register and the spill/reload pseudos alike.
static bool isVectorMemory(const MachineInstr &MI, const TargetInstrInfo &TII) {
  if (!MI.mayLoadOrStore() || MI.isInlineAsm())
    return false;
  const MCInstrDesc &Desc = MI.getDesc();
  for (unsigned I = 0, E = Desc.getNumOperands(); I != E; ++I)
    if (const TargetRegisterClass *RC = TII.getRegClass(Desc, I))
      if (RISCVRegisterInfo::isRVVRegClass(RC))
        return true;
  return false;
}

// Returns why \p MI may not run on this machine, or an empty string.
static std::string whyRefused(const MachineInstr &MI,
                              const MachineFrameInfo &MFI) {
  if (RISCVInstrInfo::isIndexedVectorMemory(MI))
    return "uses an indexed address, which takes no lane bank offset";
  if (RISCVInstrInfo::isFaultOnlyFirstLoad(MI))
    return "is a fault-only-first load, which takes no lane bank offset";
  if (MI.memoperands_empty())
    return "has no memory operand to say which memory it reaches";
  for (const MachineMemOperand *MMO : MI.memoperands()) {
    if (MMO->getAddrSpace() == RISCVAS::SPAD)
      continue;
    if (const auto *FS = dyn_cast_or_null<FixedStackPseudoSourceValue>(
            MMO->getPseudoValue())) {
      if (MFI.isSpillSlotObjectIndex(FS->getFrameIndex()))
        continue;
      return "reaches stack object " + std::to_string(FS->getFrameIndex()) +
             ", which is not a spill slot";
    }
    return "reaches address space " + std::to_string(MMO->getAddrSpace()) +
           ", not the scratchpad (address space " +
           std::to_string(RISCVAS::SPAD) + ")";
  }
  return "";
}

bool RISCVSpadVectorMemoryCheck::runOnMachineFunction(MachineFunction &MF) {
  const auto &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasSpadVectorMemory())
    return false;

  const TargetInstrInfo &TII = *ST.getInstrInfo();
  const MachineFrameInfo &MFI = MF.getFrameInfo();
  const Function &F = MF.getFunction();
  for (const MachineBasicBlock &MBB : MF) {
    for (const MachineInstr &MI : MBB) {
      if (!isVectorMemory(MI, TII))
        continue;
      std::string Why = whyRefused(MI, MFI);
      if (Why.empty())
        continue;
      std::string Msg = "spad-vector-memory: vector memory instruction " +
                        TII.getName(MI.getOpcode()).str() + " " + Why;
      F.getContext().diagnose(
          DiagnosticInfoUnsupported(F, Msg, MI.getDebugLoc()));
    }
  }
  return false;
}
