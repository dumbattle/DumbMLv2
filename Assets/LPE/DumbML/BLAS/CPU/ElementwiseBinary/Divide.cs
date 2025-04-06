namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        public static void Divide(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, FloatCPUTensorBuffer output) {
            Computation.Forward(a, b, output, new ElementwiseBinaryJob.Job<_Div, float, float, float>());
        }

        struct _Div : ElementwiseBinaryJob.IImplementation<float, float, float> {
            public float Forward(float l, float r) {
                return l / r;
            }
        }
    }

}