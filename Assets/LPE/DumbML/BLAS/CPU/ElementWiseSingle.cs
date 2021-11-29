﻿using System;

namespace DumbML.BLAS.CPU {
    public static class ElementWiseSingle {
        public static void Copy(CPUTensorBuffer input, CPUTensorBuffer dest) {
            if (!ShapeUtility.SameShape(input.shape, dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }

            for (int i = 0; i < input.size; i++) {
                dest.buffer[i] = input.buffer[i];
            }
        }
        public static void Sqr(CPUTensorBuffer input, CPUTensorBuffer dest) {
            if (!ShapeUtility.SameShape(input.shape, dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }

            for (int i = 0; i < input.size; i++) {
                var x = input.buffer[i];
                dest.buffer[i] = x * x;
            }
        }
    }
}