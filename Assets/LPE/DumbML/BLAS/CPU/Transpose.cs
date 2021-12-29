using System;
using System.Collections.Generic;
using LPE;
using Unity.Jobs;
using Unity.Collections;


namespace DumbML.BLAS.CPU {
    public static class Transpose {
        public static void Compute(FloatCPUTensorBuffer src, int[] perm, FloatCPUTensorBuffer dest) {
            // TODO - check shape

            int[] strides = Utils.GetIntArr();
            GetStrides(src.shape, strides);


            var j = new TransposeJob(src, perm, dest, strides);
            var h = j.Schedule(src.size, 1);
            h.Complete();

            j.result.CopyTo(dest.buffer);
            j.Dispose();
            Utils.Return(strides);
        }
        static void GetStrides(int[] shape, int[] result) {
            int stride = 1;

            for (int i = shape.Length - 1; i >= 0; i--) {
                result[i] = stride;

                int dimSize = shape[i];
                stride *= dimSize;
            }
        }
    }

    public struct TransposeJob : IJobParallelFor {
        public NativeArray<float> result;

        [ReadOnly]
        NativeArray<float> src;
        [ReadOnly]
        NativeArray<int> srcShape;
        [ReadOnly]
        NativeArray<int> perm;
        [ReadOnly]
        NativeArray<int> strides;

        int size;
        int rank;

        public TransposeJob(FloatCPUTensorBuffer src, int[] perm, FloatCPUTensorBuffer dest, int[] strides) {
            this.src = new NativeArray<float>(src.buffer, Allocator.TempJob);
            result = new NativeArray<float>(dest.buffer, Allocator.TempJob);

            srcShape = new NativeArray<int>(src.shape, Allocator.TempJob);
            this.perm = new NativeArray<int>(perm, Allocator.TempJob);
            this.strides = new NativeArray<int>(strides, Allocator.TempJob);

            size = src.size;
            rank = src.Rank();
        }


        public void Execute(int i) {
            int stride = size;
            int remaining = i;
            int offset = 0;

            for (int axis = 0; axis < rank; axis++) {
                int dimSize = srcShape[perm[axis]];
                stride /= dimSize;

                int indCount = remaining / stride;
                remaining = remaining % stride;

                offset += indCount * strides[perm[axis]];
            }

            result[i] = src[offset];
        }

        public void Dispose() {
            result.Dispose();
            src.Dispose();
            srcShape.Dispose();
            perm.Dispose();
            strides.Dispose();
        }
    }
}