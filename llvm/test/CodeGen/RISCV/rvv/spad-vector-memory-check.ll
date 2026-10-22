; Under +spad-vector-memory a vector memory instruction may reach only address
; space 1 (the scratchpad) or a spill slot of its own frame, and never through
; an indexed or fault-only-first address. llc refuses anything else.
;
; RUN: rm -rf %t && split-file %s %t
;
; Scratchpad vector accesses and vector spills are accepted.
; RUN: llc -mtriple=riscv64 -mattr=+v,+spad-vector-memory -verify-machineinstrs \
; RUN:   < %t/spad.ll | FileCheck %s --check-prefix=SPAD
; A vector spill slot is sized in the scratchpad, so it needs the exact VLEN.
; RUN: llc -mtriple=riscv64 -mattr=+v,+spad-vector-memory -verify-machineinstrs \
; RUN:   -riscv-v-vector-bits-max=128 < %t/spill.ll | FileCheck %s --check-prefix=SPILL
;
; Vector accesses to address space 0 are refused, at -O3 and at -O0.
; RUN: not llc -mtriple=riscv64 -mattr=+v,+spad-vector-memory \
; RUN:   < %t/dram-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DRAM-LOAD
; RUN: not llc -mtriple=riscv64 -mattr=+v,+spad-vector-memory -O0 \
; RUN:   < %t/dram-load.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DRAM-LOAD
; RUN: not llc -mtriple=riscv64 -mattr=+v,+spad-vector-memory \
; RUN:   < %t/dram-store.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=DRAM-STORE
;
; An indexed access is refused even into the scratchpad.
; RUN: not llc -mtriple=riscv64 -mattr=+v,+spad-vector-memory \
; RUN:   < %t/gather.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=GATHER
;
; A stack temporary that is not a spill slot is refused.
; RUN: not llc -mtriple=riscv64 -mattr=+v,+spad-vector-memory \
; RUN:   < %t/stack-temp.ll -o /dev/null 2>&1 | FileCheck %s --check-prefix=STACK-TEMP
;
; Without the feature nothing is refused.
; RUN: llc -mtriple=riscv64 -mattr=+v -verify-machineinstrs < %t/dram-load.ll \
; RUN:   | FileCheck %s --check-prefix=STD-LOAD
; RUN: llc -mtriple=riscv64 -mattr=+v -verify-machineinstrs < %t/dram-store.ll \
; RUN:   | FileCheck %s --check-prefix=STD-STORE
; RUN: llc -mtriple=riscv64 -mattr=+v -verify-machineinstrs < %t/gather.ll \
; RUN:   | FileCheck %s --check-prefix=STD-GATHER
; RUN: llc -mtriple=riscv64 -mattr=+v -verify-machineinstrs < %t/stack-temp.ll \
; RUN:   | FileCheck %s --check-prefix=STD-STACK-TEMP

; SPAD-LABEL: spad_copy:
; SPAD:         vle32.v v8, (a0)
; SPAD:         vse32.v v8, (a1)
; SPILL-LABEL: spad_copy_across_a_spill:
; SPILL:         auipc [[A:[a-z0-9]+]], %got_pcrel_hi(spad_copy_across_a_spill.vspill)
; SPILL:         vs2r.v v8, ([[A]])
; SPILL:         vl2r.v v8, ([[A]])

; DRAM-LOAD: error: {{.*}} in function dram_load {{.*}}: spad-vector-memory: vector memory instruction PseudoVLE32_V_M2 reaches address space 0, not the scratchpad (address space 1)
; DRAM-STORE: error: {{.*}} in function dram_store {{.*}}: spad-vector-memory: vector memory instruction PseudoVSE32_V_M2 reaches address space 0, not the scratchpad (address space 1)
; GATHER: error: {{.*}} in function spad_gather {{.*}}: spad-vector-memory: vector memory instruction PseudoVLUXEI64_V_M4_M2_MASK uses an indexed address, which takes no lane bank offset
; STACK-TEMP: error: {{.*}} in function extract_at {{.*}}: spad-vector-memory: vector memory instruction {{PseudoVSE32_V_M8|VS8R_V}} reaches stack object {{[0-9]+}}, which is not a spill slot

; STD-LOAD-LABEL: dram_load:
; STD-LOAD:         vle32.v v8, (a0)
; STD-STORE-LABEL: dram_store:
; STD-STORE:         vse32.v v8, (a1)
; STD-GATHER-LABEL: spad_gather:
; STD-GATHER:         vluxei64.v
; STD-STACK-TEMP-LABEL: extract_at:
; STD-STACK-TEMP:         vs{{e32.v|8r.v}} v8, (

;--- spad.ll
define void @spad_copy(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
  %v = load <8 x float>, ptr addrspace(1) %src, align 32
  store <8 x float> %v, ptr addrspace(1) %dst, align 32
  ret void
}

;--- spill.ll
define void @spad_copy_across_a_spill(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
  %v = load <8 x float>, ptr addrspace(1) %src, align 32
  call void asm sideeffect "",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  store <8 x float> %v, ptr addrspace(1) %dst, align 32
  ret void
}

;--- dram-load.ll
define void @dram_load(ptr %src, ptr addrspace(1) %dst) {
  %v = load <8 x float>, ptr %src, align 32
  store <8 x float> %v, ptr addrspace(1) %dst, align 32
  ret void
}

;--- dram-store.ll
define void @dram_store(ptr addrspace(1) %src, ptr %dst) {
  %v = load <8 x float>, ptr addrspace(1) %src, align 32
  store <8 x float> %v, ptr %dst, align 32
  ret void
}

;--- gather.ll
define <8 x float> @spad_gather(<8 x ptr addrspace(1)> %ptrs, <8 x i1> %m) {
  %v = call <8 x float> @llvm.masked.gather.v8f32.v8p1(<8 x ptr addrspace(1)> %ptrs, i32 4, <8 x i1> %m, <8 x float> poison)
  ret <8 x float> %v
}

;--- stack-temp.ll
define i32 @extract_at(ptr addrspace(1) %src, i64 %i) {
  %v = load <256 x i32>, ptr addrspace(1) %src, align 32
  %e = extractelement <256 x i32> %v, i64 %i
  ret i32 %e
}
