; The regalloc SpillReloadCopies remark counts, besides every spill and reload,
; those whose register is of a class whose every legal type is a vector
; (NumVectorSpills, NumVectorReloads), and the register units those vector
; spills take (VectorSpilledRegisterUnits), so a vector spill is told apart from
; a scalar one without reading the assembly. RISC-V allocates vector registers
; first, and each allocation reports the whole function, so a vector spill is
; counted in both remarks. A scalar spill (GPR, which also holds the P
; extension's packed vectors) carries none of the keys. The per-line
; BlockSpillReloadCopies remarks carry the same keys.
; RUN: llc -mtriple=riscv64 -mattr=+v -O3 %s -o /dev/null \
; RUN:   -pass-remarks-output=%t.yaml -pass-remarks-filter=regalloc
; RUN: FileCheck %s --input-file=%t.yaml

; CHECK:      Name: BlockSpillReloadCopies
; CHECK-NEXT: Function: vector_spill
; CHECK-NOT:  generated in
; CHECK:      NumVectorSpills: '1'
; CHECK-NOT:  generated in
; CHECK:      VectorSpilledRegisterUnits: '2'
; CHECK-NOT:  generated in
; CHECK:      NumVectorReloads: '1'
; CHECK:      Name: SpillReloadCopies
; CHECK-NEXT: Function: vector_spill
; CHECK-NOT:  generated in
; CHECK:      NumVectorSpills: '1'
; CHECK-NOT:  generated in
; CHECK:      VectorSpilledRegisterUnits: '2'
; CHECK-NOT:  generated in
; CHECK:      NumVectorReloads: '1'
; CHECK:      Name: SpillReloadCopies
; CHECK-NEXT: Function: vector_spill
; CHECK-NOT:  generated in
; CHECK:      NumSpills: '1'
; CHECK-NOT:  generated in
; CHECK:      NumReloads: '1'
; CHECK-NOT:  generated in
; CHECK:      NumVectorSpills: '1'
; CHECK-NOT:  generated in
; CHECK:      VectorSpilledRegisterUnits: '2'
; CHECK-NOT:  generated in
; CHECK:      NumVectorReloads: '1'
; CHECK:      Name: BlockSpillReloadCopies
; CHECK-NEXT: Function: scalar_spill
; CHECK-NOT:  Vector
; CHECK:      Name: SpillReloadCopies
; CHECK-NEXT: Function: scalar_spill
; CHECK-NOT:  generated in
; CHECK:      NumSpills: '2'
; CHECK-NOT:  Vector
; CHECK:      generated in function

define void @vector_spill(ptr %src, ptr %dst) {
  %v = load <8 x float>, ptr %src, align 32
  call void asm sideeffect "",
    "~{v0},~{v1},~{v2},~{v3},~{v4},~{v5},~{v6},~{v7},~{v8},~{v9},~{v10},~{v11},~{v12},~{v13},~{v14},~{v15},~{v16},~{v17},~{v18},~{v19},~{v20},~{v21},~{v22},~{v23},~{v24},~{v25},~{v26},~{v27},~{v28},~{v29},~{v30},~{v31}"()
  store <8 x float> %v, ptr %dst, align 32
  ret void
}

define void @scalar_spill(ptr %src, ptr %dst) {
  %x = load i64, ptr %src, align 8
  call void asm sideeffect "",
    "~{x1},~{x5},~{x6},~{x7},~{x8},~{x9},~{x10},~{x11},~{x12},~{x13},~{x14},~{x15},~{x16},~{x17},~{x18},~{x19},~{x20},~{x21},~{x22},~{x23},~{x24},~{x25},~{x26},~{x27},~{x28},~{x29},~{x30},~{x31}"()
  store i64 %x, ptr %dst, align 8
  ret void
}
