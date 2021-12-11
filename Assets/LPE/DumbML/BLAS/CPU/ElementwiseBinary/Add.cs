using LPE;
using System;

namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        public static void Add(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, FloatCPUTensorBuffer output) {
            Computation<AddDelegateCache>.Forward(a, b, output);
        }


        class AddDelegateCache : ComputeDelegateCache {
            public override void Forward(float[] a, float[] b, float[] d, int startL, int startR, int startD, int stride) {
                for (int i = 0; i < stride; i++) {
                    d[startD + i] = a[startL + i] + b[startR + i];
                }
            }
        }
    }
}