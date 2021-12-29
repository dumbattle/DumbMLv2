using LPE;
using System;
using Unity.Jobs;

namespace DumbML.BLAS.CPU {
    public static class MatrixMult {
        public static void Compute(FloatCPUTensorBuffer l, FloatCPUTensorBuffer r, FloatCPUTensorBuffer dest,
                                   bool transposeL = false, bool transposeR = false) {
            var (numBatchesL, numBatchesR) = CheckShapes(l, r, dest, transposeL, transposeR);

            var j = new MatrixMultJob(l, r, dest, transposeL, transposeR, numBatchesL, numBatchesR);

            var h = j.Schedule(dest.size, 1);
            h.Complete();
            j.result.CopyTo(dest.buffer);
            j.Dispose();
        }

        private static (int, int) CheckShapes(FloatCPUTensorBuffer l, FloatCPUTensorBuffer r, FloatCPUTensorBuffer dest, bool tl, bool tr) {
            int ldims = l.Rank();
            int rdims = r.Rank();
            int ddims = UnityEngine.Mathf.Max(ldims, rdims);

            // check ranks > 2
            if (ldims < 2) {
                throw new ArgumentException($"MatrixMult requires tensors to have dimension of at least 2. Got shape: {l.shape.ContentString()}");
            }
            if (rdims < 2) {
                throw new ArgumentException($"MatrixMult requires tensors to have dimension of at least 2. Got shape: {r.shape.ContentString()}");
            }

            // dest has correct rank
            if (dest.Rank() != ddims) {
                throw new InvalidOperationException($"Output Tensors do not have correcct rank\n  Expected{ddims}\n  Got:{dest.shape.ContentString()}");
            }


            // check leading dimensions
            // determine number of batches
            int numBatchesL = 1;
            int numBatchesR = 1;

            // can't start from 0 because l and r might have different ranks (ie. 1 of them might have implicit leading dimensions)
            // instead we use distancce from end to get dimension
            // negative = implied dimension of [1]
            // stop at 2 because we 2 dimensions are for matmult
            for (int i = ddims; i > 2; i--) {
                int dimSize = -1;

                int li = ldims - i;
                int ri = rdims - i;
                int di = ddims - i;

                int lsize = li >= 0 ? l.shape[li] : 1;
                int rsize = ri >= 0 ? r.shape[ri] : 1;

                // same
                if (rsize == lsize) {
                    dimSize = rsize;
                }
                // left is broadcastable to right
                else if (lsize == 1) {
                    dimSize = rsize;
                }

                // right is broadcastable to left
                else if (rsize == 1) {
                    dimSize = lsize;
                }

                // not compatable
                if (dimSize == -1) {
                    throw new InvalidOperationException(
                        $"Input Tensors do not have compatable leading dimensions for MatrixMult: {l.shape.ContentString()}, {r.shape.ContentString()}"
                    );
                }

                // dest doesnt have correct shape
                if (dimSize != dest.shape[di]) {
                    throw new InvalidOperationException(
                        $"Destination tensor does not have compatable batch dimensions: {dest.shape.ContentString()} Expected '{dimSize}' at index '{di}'"
                    );

                }

                numBatchesL *= lsize;
                numBatchesR *= rsize;
            }

            // check shape compatability

            int lx = l.shape[ldims - (tl ? 1 : 2)];
            int ly = l.shape[ldims - (tl ? 2 : 1)];
            int rx = r.shape[rdims - (tr ? 1 : 2)];
            int ry = r.shape[rdims - (tr ? 2 : 1)];

            if (ly != rx) {
                throw new InvalidOperationException($"Tensors do not have compatible dimensions: {l.shape.ContentString()}, {r.shape.ContentString()}");
            }
            if (dest.shape[ddims - 2] != lx || dest.shape[ddims - 1] != ry) {
                throw new InvalidOperationException($"Output Tensor does not have correct shape - Expected: [ .., {lx}, {ry} ] Got: {dest.shape.ContentString()}");

            }

            return (numBatchesL, numBatchesR);
        }
    }
}