using System;
using Unity.Jobs;


namespace DumbML.BLAS.CPU {
    public static class Cast {
        public static void Run(CPUTensorBuffer<bool> input, CPUTensorBuffer<float> dest) {
            if (!ShapeUtility.SameShape(input.shape, dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }

            var j = new CastJobs.BoolToFloat(input, dest);
            var h = j.Schedule(input.size, 64);
            h.Complete();
        }
        public static void Run(CPUTensorBuffer<float> input, CPUTensorBuffer<bool> dest) {
            if (!ShapeUtility.SameShape(input.shape, dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }

            var j = new CastJobs.FloatToBool(input, dest);
            var h = j.Schedule(input.size, 64);
            h.Complete();
        }
        public static void Run(CPUTensorBuffer<bool> input, CPUTensorBuffer<int> dest) {
            if (!ShapeUtility.SameShape(input.shape, dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }

            var j = new CastJobs.BoolToInt(input, dest);
            var h = j.Schedule(input.size, 64);
            h.Complete();
        }
        public static void Run(CPUTensorBuffer<int> input, CPUTensorBuffer<bool> dest) {
            if (!ShapeUtility.SameShape(input.shape, dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }

            var j = new CastJobs.IntToBool(input, dest);
            var h = j.Schedule(input.size, 64);
            h.Complete();
        }
        public static void Run(CPUTensorBuffer<int> input, CPUTensorBuffer<float> dest) {
            if (!ShapeUtility.SameShape(input.shape, dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }

            var j = new CastJobs.IntToFloat(input, dest);
            var h = j.Schedule(input.size, 64);
            h.Complete();
        }
        public static void Run(CPUTensorBuffer<float> input, CPUTensorBuffer<int> dest) {
            if (!ShapeUtility.SameShape(input.shape, dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }

            var j = new CastJobs.FloatToInt(input, dest);
            var h = j.Schedule(input.size, 64);
            h.Complete();
        }
    }

}