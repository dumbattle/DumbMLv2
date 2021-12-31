namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        public static void LessThanOrEquals(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, BoolCPUTensorBuffer output) {
            Computation.Forward(a, b, output, new ElementwiseBinaryJob.Job<_LessThanOrEquals, float, float, bool>());
        }
        struct _LessThanOrEquals : ElementwiseBinaryJob.IImplementation<float, float, bool> {
            public bool Forward(float l, float r) {
                return l <= r;
            }
        }
    }

}