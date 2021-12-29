using Unity.Jobs;
using Unity.Collections;


namespace DumbML.BLAS.CPU {
    public struct SetValueJob : IJobParallelFor {
        public NativeArray<float> buffer;
        float val;

        public SetValueJob(FloatCPUTensorBuffer buffer, float v) {
            this.buffer = buffer.buffer;
            val = v;
        }

        public void Execute(int index) {
            buffer[index] = val;
        }
    }
}