using System;

namespace DumbML.BLAS.CPU {
    public static class ElementWiseFloatParam {
        public static void Add(FloatCPUTensorBuffer input, FloatCPUTensorBuffer dest, float val) {
            if (!input.shape.CompareContents(dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }

            for (int i = 0; i < input.size; i++) {
                dest.buffer[i] = input.buffer[i] + val;
            }
        }
        public static void Subtract(FloatCPUTensorBuffer input, FloatCPUTensorBuffer dest, float val) {
            Add(input, dest, -val);
        }
        public static void Multiply(FloatCPUTensorBuffer input, FloatCPUTensorBuffer dest, float val) {
            if (!input.shape.CompareContents(dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }

            for (int i = 0; i < input.size; i++) {
                dest.buffer[i] = input.buffer[i] * val;
            }
        }
    }
}