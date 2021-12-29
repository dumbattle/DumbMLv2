using Unity.Collections;
namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        public static void Multiply(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, FloatCPUTensorBuffer output) {
            Computation<_Multiply>.Forward(a, b, output, new ElementwiseBinaryJob.Job<_Multiply>());
        }

        struct _Multiply : ElementwiseBinaryJob.IImplementation {
            public float Forward(float l, float r) {
                return l * r;
            }
        }
    }
}
