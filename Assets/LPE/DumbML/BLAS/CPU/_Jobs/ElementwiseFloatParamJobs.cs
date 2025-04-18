﻿using Unity.Jobs;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

namespace DumbML.BLAS.CPU {
    public static class ElementwiseFloatParamJobs {
        public struct Add : IJobParallelFor {
            NativeArray<float> src;
            [NativeDisableContainerSafetyRestriction]
            NativeArray<float> result;
            float p;
            public Add(FloatCPUTensorBuffer src, float p, FloatCPUTensorBuffer dest) {
                this.src = src.buffer;
                result = dest.buffer;
                this.p = p;
            }
            public void Execute(int i) {
                result[i] = src[i] + p;
            }
        }

        public struct Multiply : IJobParallelFor {
            NativeArray<float> src;
            [NativeDisableContainerSafetyRestriction]
            NativeArray<float> result;
            float p;
            public Multiply(FloatCPUTensorBuffer src, float p, FloatCPUTensorBuffer dest) {
                this.src = src.buffer;
                result = dest.buffer;
                this.p = p;
            }
            public void Execute(int i) {
                result[i] = src[i] * p;
            }
        }

        public struct Max : IJobParallelFor {
            NativeArray<float> src;
            [NativeDisableContainerSafetyRestriction]
            NativeArray<float> result;
            float p;
            public Max(FloatCPUTensorBuffer src, float p, FloatCPUTensorBuffer dest) {
                this.src = src.buffer;
                result = dest.buffer;
                this.p = p;
            }
            public void Execute(int i) {
                var s = src[i];
                result[i] = src[i] < p ? p : s;
            }
        }

        public struct Min : IJobParallelFor {
            NativeArray<float> src;
            [NativeDisableContainerSafetyRestriction]
            NativeArray<float> result;
            float p;
            public Min(FloatCPUTensorBuffer src, float p, FloatCPUTensorBuffer dest) {
                this.src = src.buffer;
                result = dest.buffer;
                this.p = p;
            }
            public void Execute(int i) {
                var s = src[i];
                result[i] = src[i] > p ? p : s;
            }
        }
    }
}