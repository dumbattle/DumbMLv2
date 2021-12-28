using Unity.Jobs;
using Unity.Collections;


namespace DumbML.BLAS.CPU {
    public static class ElementWiseSingleJobs {
        public struct Copy : IJobParallelFor {
            NativeArray<float> src;
            public NativeArray<float> result;

            public Copy(FloatCPUTensorBuffer src, FloatCPUTensorBuffer dest) {
                this.src = new NativeArray<float>(src.buffer, Allocator.TempJob);
                this.result = new NativeArray<float>(dest.buffer, Allocator.TempJob);
            }
            public void Execute(int index) {
                result[index] = src[index];
            }

            public void Dispose() {
                src.Dispose();
                result.Dispose();
            }
        }
        public struct Sqr : IJobParallelFor {
            NativeArray<float> src;
            public NativeArray<float> result;

            public Sqr(FloatCPUTensorBuffer src, FloatCPUTensorBuffer dest) {
                this.src = new NativeArray<float>(src.buffer, Allocator.TempJob);
                this.result = new NativeArray<float>(dest.buffer, Allocator.TempJob);
            }
            public void Execute(int index) {
                var v = src[index];
                result[index] = v * v;
            }

            public void Dispose() {
                src.Dispose();
                result.Dispose();
            }
        }
    }
}