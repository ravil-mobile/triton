// RUN: triton-opt %s -split-input-file -triton-amdgpu-refine-ops='arch=gfx942' | FileCheck %s

// CHECK-LABEL: tt.func public @exp_kernel([[VALUE_0:%.*]]
// CHECK-DAG: [[VALUE_1:%.*]] = amdgpu.extract_slice [[VALUE_0]] [0, 0]
// CHECK-DAG: [[VALUE_2:%.*]] = math.exp2 [[VALUE_1]]
// CHECK-DAG: [[VALUE_3:%.*]] = amdgpu.extract_slice [[VALUE_0]] [0, 16]
// CHECK-DAG: [[VALUE_4:%.*]] = math.exp2 [[VALUE_3]]
// CHECK-DAG: [[VALUE_5:%.*]] = amdgpu.extract_slice [[VALUE_0]] [64, 0]
// CHECK-DAG: [[VALUE_6:%.*]] = math.exp2 [[VALUE_5]]
// CHECK-DAG: [[VALUE_7:%.*]] = amdgpu.extract_slice [[VALUE_0]] [64, 16]
// CHECK-DAG: [[VALUE_8:%.*]] = math.exp2 [[VALUE_7]]
// CHECK-DAG: [[VALUE_9:%.*]] = amdgpu.concat [[VALUE_2]], [[VALUE_4]], [[VALUE_6]], [[VALUE_8]]
// CHECK-DAG: tt.return [[VALUE_9]]
#blocked = #ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [8, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "hip:gfx942", "ttg.threads-per-warp" = 64 : i32} {
  tt.func public @exp_kernel(%arg0: tensor<128x32xf32, #blocked>) -> tensor<128x32xf32, #blocked> attributes {noinline = false} {
    amdgpu.instruction_sched_hint {isBufferLoadsAEnabled = false, isBufferLoadsBEnabled = false, numDsReadsA = #amdgpu.InstCounter<0, none>, numDsReadsB = #amdgpu.InstCounter<0, none>, numDsWritesA = #amdgpu.InstCounter<0, none>, numDsWritesB = #amdgpu.InstCounter<0, none>, numGlobalLoadsA = #amdgpu.InstCounter<0, none>, numGlobalLoadsB = #amdgpu.InstCounter<0, none>, numMMAs = #amdgpu.InstCounter<0, none>, variant = #amdgpu.SchedHintVariant<refine_ops>}
    %0 = math.exp2 %arg0 : tensor<128x32xf32, #blocked>
    tt.return %0 : tensor<128x32xf32, #blocked>
  }
}
