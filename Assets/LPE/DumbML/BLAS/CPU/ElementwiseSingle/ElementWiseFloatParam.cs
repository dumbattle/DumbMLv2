using System;
using Unity.Jobs;


namespace DumbML.BLAS.CPU {
    public static class ElementWiseFloatParam {
        public static void Add(FloatCPUTensorBuffer input, FloatCPUTensorBuffer dest, float val) {
            if (!input.shape.CompareContents(dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }
            var j = new ElementwiseFloatParamJobs.Add(input, val, dest);
            var h = j.Schedule(input.size, 64);
            h.Complete();
        }
        public static void Subtract(FloatCPUTensorBuffer input, FloatCPUTensorBuffer dest, float val) {
            Add(input, dest, -val);
        }
        public static void Multiply(FloatCPUTensorBuffer input, FloatCPUTensorBuffer dest, float val) {
            if (!input.shape.CompareContents(dest.shape)) {
                throw new InvalidOperationException($"Destination tensor does not have same shape as input: {input.shape.ContentString()}, {dest.shape.ContentString()}");
            }

            var j = new ElementwiseFloatParamJobs.Multiply(input, val, dest);
            var h = j.Schedule(input.size, 64);
            h.Complete();
        }




       

    }
}