#include <metal_stdlib>
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp;
using namespace mpp::tensor_ops;
kernel void k(device half *a [[buffer(1)]], device half *b [[buffer(2)]],
              device float *out [[buffer(3)]]) {
  tensor<device half, dextents<int,2>, tensor_inline>
    tA(a, dextents<int,2>(64,32), array<int,2>{1,64});
  tensor<device half, dextents<int,2>, tensor_inline>
    tB(b, dextents<int,2>(32,64), array<int,2>{1,32});
  tensor<device float, dextents<int,2>, tensor_inline>
    tC(out, dextents<int,2>(32,32), array<int,2>{1,112});
  constexpr auto desc = matmul2d_descriptor(32,32,64,false,false,false,
                                           matmul2d_descriptor::mode::multiply);
  matmul2d<desc, execution_simdgroups<1>> op;
  auto sA = tA.slice<64,32>(0,0);
  auto sB = tB.slice<32,64>(0,0);
  auto sC = tC.slice<32,32>(0,0);
  op.run(sA,sB,sC);
}
