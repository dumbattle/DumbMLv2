using Unity.Jobs;
using Unity.Collections;


namespace DumbML.BLAS.CPU {
    public static class SetValues {
        public static void Run(FloatCPUTensorBuffer buffer, float v) {
            var j = new SetValueJob(buffer, v);
            var h = j.Schedule(buffer.size, 64);
            h.Complete();
            j.buffer.CopyTo(buffer.buffer);
            j.Dispose();
        }

        public static void One(FloatCPUTensorBuffer input) {
            Run(input, 1);
        }

        public static void Zero(FloatCPUTensorBuffer input) {
            Run(input, 0);
        }
    }


    public struct SetValueJob : IJobParallelFor {
        public NativeArray<float> buffer;
        float val;

        public SetValueJob(FloatCPUTensorBuffer buffer, float v) {
            this.buffer = new NativeArray<float>(buffer.buffer, Allocator.TempJob);
            val = v;
        }

        public void Execute(int index) {
            buffer[index] = val;
        }

        public void Dispose() {
            buffer.Dispose();
        }
    }
}