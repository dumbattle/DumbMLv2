using Unity.Jobs;
using Unity.Collections;


namespace DumbML.BLAS.CPU {
    public static class ElementWiseSingleJobs {
        public struct Copy : IJobParallelFor {
            NativeArray<float> src;
            public NativeArray<float> result;

            public Copy(FloatCPUTensorBuffer src, FloatCPUTensorBuffer dest) {
                this.src = src.buffer;
                result = dest.buffer;
            }
            public void Execute(int index) {
                result[index] = src[index];
            }
        }
        public struct Exp : IJobParallelFor {
            NativeArray<float> src;
            public NativeArray<float> result;

            public Exp(FloatCPUTensorBuffer src, FloatCPUTensorBuffer dest) {
                this.src = src.buffer;
                result = dest.buffer;
            }
            public void Execute(int index) {
                result[index] = UnityEngine.Mathf.Exp(src[index]);
            }
        }
        public struct ReLU : IJobParallelFor {
            NativeArray<float> src;
            public NativeArray<float> result;

            public ReLU(FloatCPUTensorBuffer src, FloatCPUTensorBuffer dest) {
                this.src = src.buffer;
                result = dest.buffer;
            }
            public void Execute(int index) {
                var v = src[index];
                result[index] = v > 0 ? v : 0;
            }
        }
        public struct Sqr : IJobParallelFor {
            NativeArray<float> src;
            public NativeArray<float> result;

            public Sqr(FloatCPUTensorBuffer src, FloatCPUTensorBuffer dest) {
                this.src = src.buffer;
                result = dest.buffer;
            }
            public void Execute(int index) {
                var v = src[index];
                result[index] = v * v;
            }
        }
    }

}