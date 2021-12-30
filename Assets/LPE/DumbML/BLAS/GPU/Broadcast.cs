using UnityEngine;
using System;


namespace DumbML.BLAS.GPU {
    public static class Broadcast {
        public static void Compute(FloatGPUTensorBuffer input, int[] shape, FloatGPUTensorBuffer output) {
            ValidateShapes(input, shape, output);
            ComputeShader shader = Kernels.broadcast;
            int kernelID = shader.FindKernel("Broadcast");

            shader.SetBuffer(kernelID, "input", input.buffer);
            shader.SetBuffer(kernelID, "output", output.buffer);
            shader.SetInts("srcShape", input.shape);
            shader.SetInts("shape", shape);


            shader.SetInt("isize", input.size);
            shader.SetInt("dsize", output.size);
            shader.SetInt("shapeRank", shape.Length);
            shader.SetInt("srcRank", input.Rank());

            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = output.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }

        static void ValidateShapes(FloatGPUTensorBuffer input, int[] shape, FloatGPUTensorBuffer dest) {
            if (!ShapeUtility.SameShape(shape, dest.shape)) {
                throw new ArgumentException(
                    $"Destination does not have the correct shape" +
                    $"\nExpected: {shape.ContentString()}" +
                    $"\nGot: {dest.shape.ContentString()}");
            }

            for (int i = 0; i < input.Rank(); i++) {
                int ii = input.Rank() - 1 - i;
                int si = shape.Length - 1 - i;

                bool valid =
                    input.shape[ii] == 1 // broadcasted
                    || input.shape[ii] == shape[si]; // same dim
                if (!valid) {
                    throw new ArgumentException(
                        $"Cannot broadcast Tensor" +
                        $"\nInput shape: {input.shape.ContentString()}" +
                        $"\nTarget shape: {shape.ContentString()}");
                }

            }
        }
    }
}