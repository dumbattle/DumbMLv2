using Unity.Jobs;
using Unity.Collections;


namespace DumbML.BLAS.CPU {
    public static class ElementwiseFloatParamJobs {
        public struct Add : IJobParallelFor {
            NativeArray<float> src;
            public NativeArray<float> result;
            float p;
            public Add(FloatCPUTensorBuffer src, float p, FloatCPUTensorBuffer dest) {
                this.src = new NativeArray<float>(src.buffer, Allocator.TempJob);
                this.result = new NativeArray<float>(dest.buffer, Allocator.TempJob);
                this.p = p;
            }
            public void Execute(int i) {
                result[i] = src[i] + p;
            }
            public void Dispose() {
                src.Dispose();
                result.Dispose();
            }
        }

        public struct Multiply : IJobParallelFor {
            NativeArray<float> src;
            public NativeArray<float> result;
            float p;
            public Multiply(FloatCPUTensorBuffer src, float p, FloatCPUTensorBuffer dest) {
                this.src = new NativeArray<float>(src.buffer, Allocator.TempJob);
                this.result = new NativeArray<float>(dest.buffer, Allocator.TempJob);
                this.p = p;
            }
            public void Execute(int i) {
                result[i] = src[i] * p;
            }
            public void Dispose() {
                src.Dispose();
                result.Dispose();
            }
        }
    }
}