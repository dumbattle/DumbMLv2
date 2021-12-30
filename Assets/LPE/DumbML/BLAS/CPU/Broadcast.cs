using Unity.Jobs;


namespace DumbML.BLAS.CPU {
    public static class Broadcast {
        public static void Compute(FloatCPUTensorBuffer input, int[] shape, FloatCPUTensorBuffer dest) {
            var j = new BroadcastJob(input, shape, dest);
            var h = j.Schedule(dest.size, 1);
            h.Complete();
            j.Dispose();
        }
    }
}