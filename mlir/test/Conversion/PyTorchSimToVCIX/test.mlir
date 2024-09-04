func.func @exp(%a: vector<32 x f32>, %rvl: i64) -> vector<32 x f32> {
%res = math.exp %a : vector<32 x f32>
return %res : vector<32 x f32>
}

func.func @matmul_hw_128_128_128(%A: memref<128x128xf32>, %B: memref<128x128xf32>, %C: memref<128x128xf32>) {
    linalg.matmul ins(%A, %B: memref<128x128xf32>, memref<128x128xf32>) outs(%C: memref<128x128xf32>)
    return
}

func.func @matmul_hw_512_128_128(%A: memref<512x128xf32>, %B: memref<128x128xf32>, %C: memref<512x128xf32>) {
    linalg.matmul ins(%A, %B: memref<512x128xf32>, memref<128x128xf32>) outs(%C: memref<512x128xf32>)
    return
}