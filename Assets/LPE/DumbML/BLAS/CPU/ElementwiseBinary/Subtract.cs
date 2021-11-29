namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
            public static void Subtract(CPUTensorBuffer a, CPUTensorBuffer b, CPUTensorBuffer output) {
                Computation<SubtractDelegateCache>.Forward(a, b, output);
            }


        class SubtractDelegateCache : ComputeDelegateCache {
            public override void Forward(float[] a, float[] b, float[] d, int start, int end) {
                for (int i = start; i < end; i++) {
                    d[i] = a[i] - b[i];
                }
            }
        }
    }
}