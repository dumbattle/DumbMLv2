using System;
using Unity.Jobs;


namespace DumbML.BLAS.CPU {
    public static class ElementWiseSingle {
        public static void Copy(FloatCPUTensorBuffer input, FloatCPUTensorBuffer dest, bool ignoreShape = false) {
            if (!ignoreShape && !ShapeUtility.SameShape(input.shape, dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }
            var j = new ElementWiseSingleJobs.Copy(input, dest);
            var h = j.Schedule(input.size, 64);
            h.Complete();
        }
        public static void ReLU(FloatCPUTensorBuffer input, FloatCPUTensorBuffer dest) {
            if (!ShapeUtility.SameShape(input.shape, dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }

            var j = new ElementWiseSingleJobs.ReLU(input, dest);
            var h = j.Schedule(input.size, 64);
            h.Complete();
        }
        public static void Sqr(FloatCPUTensorBuffer input, FloatCPUTensorBuffer dest) {
            if (!ShapeUtility.SameShape(input.shape, dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }

            var j = new ElementWiseSingleJobs.Sqr(input, dest);
            var h = j.Schedule(input.size, 64);
            h.Complete();
        }
    }

}