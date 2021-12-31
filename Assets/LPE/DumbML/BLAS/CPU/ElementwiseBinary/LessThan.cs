namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        public static void LessThan(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, BoolCPUTensorBuffer output) {
            Computation.Forward(a, b, output, new ElementwiseBinaryJob.Job<_LessThan, float, float, bool>());
        }
        struct _LessThan : ElementwiseBinaryJob.IImplementation<float, float, bool> {
            public bool Forward(float l, float r) {
                return l < r;
            }
        }
    }

}