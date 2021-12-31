using Unity.Jobs;
using Unity.Collections;


namespace DumbML.BLAS.CPU {
    public static class CastJobs {
        public struct BoolToFloat : IJobParallelFor {
            NativeArray<bool> src;
            NativeArray<float> result;

            public BoolToFloat(CPUTensorBuffer<bool> src, CPUTensorBuffer<float> dest) {
                this.src = src.buffer;
                result = dest.buffer;
            }
            public void Execute(int index) {
                var v = src[index];
                result[index] = v ? 1 : 0;
            }
        }
        public struct FloatToBool : IJobParallelFor {
            NativeArray<float> src;
            NativeArray<bool> result;

            public FloatToBool(CPUTensorBuffer<float> src, CPUTensorBuffer<bool> dest) {
                this.src = src.buffer;
                result = dest.buffer;
            }
            public void Execute(int index) {
                var v = src[index];
                result[index] = v != 0;
            }
        }


        public struct BoolToInt : IJobParallelFor {
            NativeArray<bool> src;
            NativeArray<int> result;

            public BoolToInt(CPUTensorBuffer<bool> src, CPUTensorBuffer<int> dest) {
                this.src = src.buffer;
                result = dest.buffer;
            }
            public void Execute(int index) {
                var v = src[index];
                result[index] = v ? 1 : 0;
            }
        }
        public struct IntToFloat : IJobParallelFor {
            NativeArray<int> src;
            NativeArray<float> result;

            public IntToFloat(CPUTensorBuffer<int> src, CPUTensorBuffer<float> dest) {
                this.src = src.buffer;
                result = dest.buffer;
            }
            public void Execute(int index) {
                var v = src[index];
                result[index] = v;
            }
        }


        public struct FloatToInt : IJobParallelFor {
            NativeArray<float> src;
            NativeArray<int> result;

            public FloatToInt(CPUTensorBuffer<float> src, CPUTensorBuffer<int> dest) {
                this.src = src.buffer;
                result = dest.buffer;
            }
            public void Execute(int index) {
                var v = src[index];
                result[index] = (int)v;
            }
        }
        public struct IntToBool : IJobParallelFor {
            NativeArray<int> src;
            NativeArray<bool> result;

            public IntToBool(CPUTensorBuffer<int> src, CPUTensorBuffer<bool> dest) {
                this.src = src.buffer;
                result = dest.buffer;
            }
            public void Execute(int index) {
                var v = src[index];
                result[index] = v != 0;
            }
        }
    }
}