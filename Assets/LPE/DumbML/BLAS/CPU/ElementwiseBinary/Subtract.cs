﻿using Unity.Collections;
namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        public static void Subtract(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, FloatCPUTensorBuffer output) {
            Computation.Forward(a, b, output, new ElementwiseBinaryJob.Job<_Subtract, float, float, float>());
        }


        struct _Subtract : ElementwiseBinaryJob.IImplementation<float, float, float> {
            public float Forward(float l, float r) {
                return l - r;
            }
        }
    }
}
