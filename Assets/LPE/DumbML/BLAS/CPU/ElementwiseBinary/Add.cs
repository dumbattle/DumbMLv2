using LPE;
using System;

namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        public static void Add(CPUTensorBuffer a, CPUTensorBuffer b, CPUTensorBuffer output) {
            Computation<AddDelegateCache>.Forward(a, b, output);
        }


        class AddDelegateCache : ComputeDelegateCache {
            public override void Forward(float[] a, float[] b, float[] d, int start, int end) {
                for (int i = start; i < end; i++) {
                    d[i] = a[i] + b[i];
                }
            }
        }
    }
}