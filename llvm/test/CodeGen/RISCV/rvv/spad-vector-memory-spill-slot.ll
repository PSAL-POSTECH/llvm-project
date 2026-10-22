; Under spad-vector-memory a vector spill slot is no stack object: each lane's
; copy lives in the function's area of the scratchpad, an internal .spad global
; in address space 1 sized from the exact VLEN. A callee-saved register the
; function leaves unused holds the area's address from the prologue on; with
; none free, each access reads the address from the GOT. Without the feature
; the slot stays on the stack.
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl256b,+spad-vector-memory \
; RUN:   -riscv-v-vector-bits-max=256 -relocation-model=pic -O3 -verify-machineinstrs \
; RUN:   %s -o - \
; RUN:   | FileCheck %s --check-prefix=SPAD
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl256b,+spad-vector-memory \
; RUN:   -riscv-v-vector-bits-max=256 -relocation-model=pic -O3 %s \
; RUN:   -stop-after=prologepilog -o - | FileCheck %s --check-prefix=MIR
; RUN: llc -mtriple=riscv64 -mattr=+v,+zvl256b -riscv-v-vector-bits-max=256 \
; RUN:   -relocation-model=pic -O3 %s -o - | FileCheck %s --check-prefix=STACK
; RUN: not llc -mtriple=riscv64 -mattr=+v,+spad-vector-memory \
; RUN:   -relocation-model=pic -O3 %s -o /dev/null 2>&1 \
; RUN:   | FileCheck %s --check-prefix=NOVLEN

; SPAD-LABEL: two_slots:
; SPAD:         sd s11, {{[0-9]+}}(sp)
; SPAD:         auipc s11, %got_pcrel_hi(two_slots.vspill)
; SPAD-NEXT:    ld s11, %pcrel_lo(.Lpcrel_hi{{[0-9]+}})(s11)
; SPAD:         vs1r.v v8, (s11)
; SPAD:         addi [[A:a[0-9]+]], s11, 32
; SPAD-NEXT:    vs2r.v v8, ([[A]])
; SPAD:         vl1r.v v8, (s11)
; SPAD:         addi [[B:a[0-9]+]], s11, 32
; SPAD-NEXT:    vl2r.v v8, ([[B]])
; SPAD:         ld s11, {{[0-9]+}}(sp)
; SPAD-NOT:     got_pcrel_hi
; SPAD-LABEL: no_free_base:
; SPAD:         auipc [[G:[a-z0-9]+]], %got_pcrel_hi(no_free_base.vspill)
; SPAD-NEXT:    ld [[G]], %pcrel_lo(.Lpcrel_hi{{[0-9]+}})([[G]])
; SPAD-NEXT:    vs1r.v v8, ([[G]])
; SPAD:         auipc [[H:[a-z0-9]+]], %got_pcrel_hi(no_free_base.vspill)
; SPAD-NEXT:    ld [[H]], %pcrel_lo(.Lpcrel_hi{{[0-9]+}})([[H]])
; SPAD-NEXT:    vl1r.v v8, ([[H]])
; SPAD-NOT:     vscale x {{[0-9]+}}-byte Folded
; SPAD:         .section .spad,"aw",@progbits
; SPAD-NEXT:    .p2align 5, 0x0
; SPAD-NEXT:  one_slot.vspill:
; SPAD-NEXT:    .zero 32
; SPAD:       two_slots.vspill:
; SPAD-NEXT:    .zero 96
; SPAD:       no_free_base.vspill:
; SPAD-NEXT:    .zero 32

; MIR: @two_slots.vspill = internal addrspace(1) global [96 x i8] zeroinitializer, section ".spad", align 32
; MIR-LABEL: name: two_slots
; MIR: VS1R_V {{.*}} :: (store (<vscale x 1 x s64>) into @two_slots.vspill, addrspace 1)
; MIR: VS2R_V {{.*}} :: (store (<vscale x 1 x s128>) into @two_slots.vspill + 32, {{.*}}addrspace 1)

; STACK-LABEL: two_slots:
; STACK:         vs1r.v v8, ({{.*}}) # vscale x 8-byte Folded Spill
; STACK-NOT:     vspill

; NOVLEN: in function one_slot {{.*}}: spad-vector-memory: a vector spill slot is sized in the scratchpad, which needs the exact VLEN

define void @one_slot(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
  %v = load <8 x float>, ptr addrspace(1) %src, align 32
  call void asm sideeffect "",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  store <8 x float> %v, ptr addrspace(1) %dst, align 32
  ret void
}

define void @two_slots(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
  %a = load <8 x float>, ptr addrspace(1) %src, align 32
  %p = getelementptr <8 x float>, ptr addrspace(1) %src, i64 1
  %b = load <16 x float>, ptr addrspace(1) %p, align 32
  call void asm sideeffect "",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  store <8 x float> %a, ptr addrspace(1) %dst, align 32
  %q = getelementptr <8 x float>, ptr addrspace(1) %dst, i64 1
  store <16 x float> %b, ptr addrspace(1) %q, align 32
  ret void
}

define void @no_free_base(ptr addrspace(1) %src, ptr addrspace(1) %dst) {
  %v = load <8 x float>, ptr addrspace(1) %src, align 32
  call void asm sideeffect "",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27}"()
  store <8 x float> %v, ptr addrspace(1) %dst, align 32
  ret void
}
