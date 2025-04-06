using System;
using Unity.Jobs;


namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseSingle {
        public static void Copy(FloatCPUTensorBuffer input, FloatCPUTensorBuffer dest, bool ignoreShape = false) {
            if (!ignoreShape && !ShapeUtility.SameShape(input.shape, dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }
            var j = new ElementWiseSingleJobs.Copy(input, dest);
            var h = j.Schedule(input.size, 64);

            h.Complete();
        }
        public static void Exp(FloatCPUTensorBuffer input, FloatCPUTensorBuffer dest, bool ignoreShape = false) {
            if (!ignoreShape && !ShapeUtility.SameShape(input.shape, dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }
            var j = new ElementWiseSingleJobs.Exp(input, dest);
            var h = j.Schedule(input.size, 64);
            h.Complete();
        }
        public static void Log(FloatCPUTensorBuffer input, FloatCPUTensorBuffer dest, bool ignoreShape = false) {
            if (!ignoreShape && !ShapeUtility.SameShape(input.shape, dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }
            var j = new ElementWiseSingleJobs.Log(input, dest);
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
        public static void Sqrt(FloatCPUTensorBuffer input, FloatCPUTensorBuffer dest) {
            if (!ShapeUtility.SameShape(input.shape, dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }

            var j = new ElementWiseSingleJobs.Sqrt(input, dest);
            var h = j.Schedule(input.size, 64);
            h.Complete();
        }
    }

}