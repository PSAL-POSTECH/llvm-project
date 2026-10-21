; RUN: llc -mtriple=riscv64 -mattr=+v -pass-remarks-missed=regalloc \
; RUN:   -o /dev/null %s 2>&1 | FileCheck %s --implicit-check-not="<unknown>"

declare void @g()

; A vector value live across a call is spilled after its def and reloaded right
; before its use: the reload is counted under the line that uses it (3), not the
; call's line (2), and the LMUL 8 group is 8 register units.
; CHECK: remark: k.py:1:0: 1 spills {{.*}} 8 register units spilled {{.*}}generated in this line's code
; CHECK-NEXT: remark: k.py:3:0: 1 reloads {{.*}}generated in this line's code
define void @reload_at_use(ptr %p) !dbg !5 {
  %a = load <vscale x 16 x i32>, ptr %p, !dbg !10
  call void @g(), !dbg !11
  %b = add <vscale x 16 x i32> %a, %a, !dbg !12
  store <vscale x 16 x i32> %b, ptr %p, !dbg !12
  ret void, !dbg !12
}

; The reload is the first instruction of its block: it is still counted under
; the line it serves, not left without a location.
; CHECK: remark: k.py:11:0: 1 spills {{.*}} 8 register units spilled {{.*}}generated in this line's code
; CHECK-NEXT: remark: k.py:13:0: 1 reloads {{.*}}generated in this line's code
define void @reload_at_block_start(ptr %p) !dbg !6 {
entry:
  %a = load <vscale x 16 x i32>, ptr %p, !dbg !20
  call void @g(), !dbg !21
  br label %next, !dbg !21
next:
  %b = add <vscale x 16 x i32> %a, %a, !dbg !22
  store <vscale x 16 x i32> %b, ptr %p, !dbg !22
  ret void, !dbg !22
}

!llvm.dbg.cu = !{!0}
!llvm.module.flags = !{!2, !3}

!0 = distinct !DICompileUnit(language: DW_LANG_C99, file: !1, emissionKind: LineTablesOnly)
!1 = !DIFile(filename: "k.py", directory: "/")
!2 = !{i32 2, !"Debug Info Version", i32 3}
!3 = !{i32 7, !"Dwarf Version", i32 5}
!4 = !DISubroutineType(types: !{})
!5 = distinct !DISubprogram(name: "reload_at_use", scope: !1, file: !1, line: 1, type: !4, unit: !0, spFlags: DISPFlagDefinition)
!6 = distinct !DISubprogram(name: "reload_at_block_start", scope: !1, file: !1, line: 10, type: !4, unit: !0, spFlags: DISPFlagDefinition)
!10 = !DILocation(line: 1, scope: !5)
!11 = !DILocation(line: 2, scope: !5)
!12 = !DILocation(line: 3, scope: !5)
!20 = !DILocation(line: 11, scope: !6)
!21 = !DILocation(line: 12, scope: !6)
!22 = !DILocation(line: 13, scope: !6)
