using Unity.Jobs;


namespace DumbML.BLAS.CPU {
    public static class SetValues {
        public static void Run(FloatCPUTensorBuffer buffer, float v) {
            var j = new SetValueJob(buffer, v);
            var h = j.Schedule(buffer.size, 64);
            h.Complete();
        }

        public static void One(FloatCPUTensorBuffer input) {
            Run(input, 1);
        }

        public static void Zero(FloatCPUTensorBuffer input) {
            Run(input, 0);
        }
    }
}