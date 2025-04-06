namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        public static void Multiply(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, FloatCPUTensorBuffer output) {
            Computation.Forward(a, b, output, new ElementwiseBinaryJob.Job<_Multiply, float, float, float>());
        }

        struct _Multiply : ElementwiseBinaryJob.IImplementation<float, float, float> {
            public float Forward(float l, float r) {
                return l * r;
            }
        }
    }
}
