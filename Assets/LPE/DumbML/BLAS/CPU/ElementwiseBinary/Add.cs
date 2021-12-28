using Unity.Collections;

namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        public static void Add(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, FloatCPUTensorBuffer output) {
            Computation<_Add>.Forward(a, b, output);
        }

        struct _Add : ElementwiseBinaryJob.IImplementation {
            public float Forward(float l, float r) {
                return l + r;
            }
        }
    }
}