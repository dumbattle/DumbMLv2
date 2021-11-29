namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
            public static void Multiply(CPUTensorBuffer a, CPUTensorBuffer b, CPUTensorBuffer output) {
                Computation<MultiplyDelegateCache>.Forward(a, b, output);
            }
        class MultiplyDelegateCache : ComputeDelegateCache {
            public override void Forward(float[] a, float[] b, float[] d, int start, int end) {
                for (int i = start; i < end; i++) {
                    d[i] = a[i] * b[i];
                }
            }
        }
    }
}