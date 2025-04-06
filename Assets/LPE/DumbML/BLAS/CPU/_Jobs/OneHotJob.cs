using Unity.Jobs;
using Unity.Collections;


namespace DumbML.BLAS.CPU {
    public struct OneHotJob<T> : IJobParallelFor where T : struct {
        [ReadOnly]
        NativeArray<int> indices;
        NativeArray<T> dest;
        int depth;
        T onval;
        T offval;


        public OneHotJob(CPUTensorBuffer<int> indices, int depth, T onval, T offval, CPUTensorBuffer<T> dest) {
            this.indices = indices.buffer;
            this.dest = dest.buffer;
            this.depth = depth;
            this.onval = onval;
            this.offval = offval;
        }

        public void Execute(int i) {
            int srcInd = i / depth;
            int offset = i % depth;

            int ind = indices[srcInd];

            T v;

            if (offset == ind) {
                v = onval;
            }
            else {
                v = offval;
            }

            dest[i] = v;
        }
    }
}
