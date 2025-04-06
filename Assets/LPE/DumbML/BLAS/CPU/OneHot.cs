using Unity.Jobs;


namespace DumbML.BLAS.CPU {
    public static class OneHot {
        public static void Compute<T>(CPUTensorBuffer<int> indices, int depth, T onval, T offval, CPUTensorBuffer<T> dest) where T : struct {
            var j = new OneHotJob<T>(indices, depth, onval, offval, dest);
            var h = j.Schedule(dest.size, 1);
            h.Complete();
        }
    }
}
