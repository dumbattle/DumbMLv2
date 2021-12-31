namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        public static void Max(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, FloatCPUTensorBuffer output) {
            Computation.Forward(a, b, output, new ElementwiseBinaryJob.Job<_Max, float, float, float>());
        }

        struct _Max : ElementwiseBinaryJob.IImplementation<float, float, float> {
            public float Forward(float l, float r) {
                return l > r ? l : r;
            }
        }
    }

}