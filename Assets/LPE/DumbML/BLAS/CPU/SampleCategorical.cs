using Unity.Jobs;
using LPE;

namespace DumbML.BLAS.CPU {
    public static class SampleCategorical {
        public static void Compute(FloatCPUTensorBuffer src, IntCPUTensorBuffer dest) {
            // Validate shape
            if (dest.Rank() != src.Rank()) {
                throw new System.ArgumentException($"Incompatible Destination shapes for SampleCategorical\nSrc {src.shape.ContentString()} \nDest {dest.shape.ContentString()}");
            }

            for (int i = 0; i < dest.Rank() - 1; i++) {
                if (dest.shape[i] != src.shape[i]) {
                    throw new System.ArgumentException($"Incompatible Destination shapes for SampleCategorical\nSrc {src.shape.ContentString()} \nDest {dest.shape.ContentString()}");
                }
            }

            if (dest.shape[dest.Rank() - 1] != 1) {
                throw new System.ArgumentException($"Invalid Destination shapes for SampleCategorical\nSrc {src.shape.ContentString()} \nDest {dest.shape.ContentString()}");
            }
            var j = new SampleCategoricalJob(src, dest);
            var h = j.Schedule(dest.size, 64);
            h.Complete();
        }
    }
}