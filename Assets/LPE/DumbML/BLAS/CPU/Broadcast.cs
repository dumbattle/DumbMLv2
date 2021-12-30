using Unity.Jobs;
using System;

namespace DumbML.BLAS.CPU {
    public static class Broadcast {
        public static void Compute(FloatCPUTensorBuffer input, int[] shape, FloatCPUTensorBuffer dest) {
            ValidateShapes(input, shape, dest);
            var j = new BroadcastJob(input, shape, dest);
            var h = j.Schedule(dest.size, 1);
            h.Complete();
            j.Dispose();
        }

        static void ValidateShapes(FloatCPUTensorBuffer input, int[] shape, FloatCPUTensorBuffer dest) {
            if (!ShapeUtility.SameShape(shape, dest.shape) ) {
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