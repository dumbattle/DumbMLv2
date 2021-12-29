using System;
using System.Collections.Generic;
using LPE;
using Unity.Jobs;


namespace DumbML.BLAS.CPU {
    public static class Transpose {
        public static void Compute(FloatCPUTensorBuffer src, int[] perm, FloatCPUTensorBuffer dest) {
            // TODO - check shape

            int[] strides = Utils.GetIntArr();
            GetStrides(src.shape, strides);


            var j = new TransposeJob(src, perm, dest, strides);
            var h = j.Schedule(src.size, 1);
            h.Complete();
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
}