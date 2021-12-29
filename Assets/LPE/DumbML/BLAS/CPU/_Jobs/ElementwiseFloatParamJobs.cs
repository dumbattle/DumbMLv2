using Unity.Jobs;
using Unity.Collections;


namespace DumbML.BLAS.CPU {
    public static class ElementwiseFloatParamJobs {
        public struct Add : IJobParallelFor {
            NativeArray<float> src;
            public NativeArray<float> result;
            float p;
            public Add(FloatCPUTensorBuffer src, float p, FloatCPUTensorBuffer dest) {
                this.src = src.buffer;
                result = dest.buffer;
                this.p = p;
            }
            public void Execute(int i) {
                result[i] = src[i] + p;
            }
        }

        public struct Multiply : IJobParallelFor {
            NativeArray<float> src;
            public NativeArray<float> result;
            float p;
            public Multiply(FloatCPUTensorBuffer src, float p, FloatCPUTensorBuffer dest) {
                this.src = src.buffer;
                result = dest.buffer;
                this.p = p;
            }
            public void Execute(int i) {
                result[i] = src[i] * p;
            }
        }
    }
}