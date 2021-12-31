﻿namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        public static void GreaterThan(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, BoolCPUTensorBuffer output) {
            Computation.Forward(a, b, output, new ElementwiseBinaryJob.Job<_GreaterThan, float, float, bool>());
        }

        struct _GreaterThan : ElementwiseBinaryJob.IImplementation<float, float, bool> {
            public bool Forward(float l, float r) {
                return l > r;
            }
        }
    }

}