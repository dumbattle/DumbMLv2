namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        public static void Min(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, FloatCPUTensorBuffer output) {
            Computation.Forward(a, b, output, new ElementwiseBinaryJob.Job<_Min, float, float, float>());
        }

        struct _Min : ElementwiseBinaryJob.IImplementation<float, float, float> {
            public float Forward(float l, float r) {
                return l < r ? l : r;
            }
        }
    }

}