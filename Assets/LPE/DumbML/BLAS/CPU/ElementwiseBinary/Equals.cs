namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        public static void Equals(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, BoolCPUTensorBuffer output) {
            Computation.Forward(a, b, output, new ElementwiseBinaryJob.Job<_Equals, float, float, bool>());
        }

        struct _Equals : ElementwiseBinaryJob.IImplementation<float, float, bool> {
            public bool Forward(float l, float r) {
                return l == r;
            }
        }
    }

}