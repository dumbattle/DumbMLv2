using UnityEngine;


namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        public static void Add(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, FloatCPUTensorBuffer output) {
            Computation.Forward(a, b, output, new ElementwiseBinaryJob.Job<_Add, float, float, float>());
        }

        struct _Add : ElementwiseBinaryJob.IImplementation<float, float, float> {
            public float Forward(float l, float r) {
                return l + r;
            }
        }
    }

}