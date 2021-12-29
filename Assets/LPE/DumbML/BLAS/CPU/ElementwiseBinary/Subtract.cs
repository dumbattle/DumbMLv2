using Unity.Collections;
namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        public static void Subtract(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, FloatCPUTensorBuffer output) {
            Computation<_Subtract>.Forward(a, b, output, new ElementwiseBinaryJob.Job<_Subtract>());
        }


        struct _Subtract : ElementwiseBinaryJob.IImplementation {
            public float Forward(float l, float r) {
                return l - r;
            }
        }
    }
}
