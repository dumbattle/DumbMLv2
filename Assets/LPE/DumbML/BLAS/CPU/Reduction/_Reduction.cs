using LPE;
using System.Threading;
using System.Collections.Generic;
using Unity.Jobs;


namespace DumbML.BLAS.CPU {
    public static partial class Reduction {
        static class Reduce<T> where T : struct, ReductionJob.IImplementation {

            public static void Compute(FloatCPUTensorBuffer src, int[] axis, FloatCPUTensorBuffer dest) {
                var j = new ReductionJob.Job<T>(src, axis, dest);

                var h = j.Schedule(dest.size, 1);

                h.Complete();
                j.result.CopyTo(dest.buffer);
                j.Dispose();
            }

        }
    }


}
