using LPE;
using System;

namespace DumbML.BLAS.CPU {
    public static class MatrixMult {
        static ObjectPool<ComputeDelegateCache> forwardIterationPool
            = new ObjectPool<ComputeDelegateCache>(() => new ComputeDelegateCache());


        public static void Compute(CPUTensorBuffer l, CPUTensorBuffer r, CPUTensorBuffer dest) {
            int numBatches = CheckShapes(l, r, dest);

            // compute
            ComputeDelegateCache forward = forwardIterationPool.Get();
            forward.left = l;
            forward.right = r;
            forward.output = dest;
            forward.maxBatches = numBatches;
            var cb = ThreadPool.ForWithCallback(0, numBatches, forward.CallForwardAction);
            cb.Wait();

            forwardIterationPool.Return(forward);
        }

        private static int CheckShapes(CPUTensorBuffer l, CPUTensorBuffer r, CPUTensorBuffer dest) {
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
            int numBatches = 1;

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

                numBatches *= dimSize;
            }

            // check shape compatability

            int lx = l.shape[ldims - 2];
            int ly = l.shape[ldims - 1];
            int rx = r.shape[rdims - 2];
            int ry = r.shape[rdims - 1];

            if (ly != rx) {
                throw new InvalidOperationException($"Tensors do not have compatible dimensions: {l.shape.ContentString()}, {r.shape.ContentString()}");
            }
            if (dest.shape[ddims - 2] != lx || dest.shape[ddims - 1] != ry) {
                throw new InvalidOperationException($"Output Tensor does not have correct shape - Expected: [ .., {lx}, {ry} ] Got: {dest.shape.ContentString()}");

            }

            return numBatches;
        }



        /// <summary>
        /// Creating delegates creates garbage (when we pass to thread pool), so we cache it in this class
        /// </summary>
        class ComputeDelegateCache {
            public CPUTensorBuffer left;
            public CPUTensorBuffer right;
            public CPUTensorBuffer output;

            public int maxBatches;
            public Action<int> CallForwardAction;

            public ComputeDelegateCache() {
                CallForwardAction = CallForward;
                //CallBackwardsAction = CallBackwards;
            }
            void CallForward(int i) {
                // cache
                int ldims = left.Rank();
                int rdims = right.Rank();
                int ddims = output.Rank();

                int lx = left.shape[ldims - 2];
                int ly = left.shape[ldims - 1];
                int rx = right.shape[rdims - 2];
                int ry = right.shape[rdims - 1];


                var lv = left.buffer;
                var rv = right.buffer;
                var dv = output.buffer;

                // check for broadcasting
                int lind = 0;
                int rind = 0;
                int remaining = i;
                int stride = maxBatches;


                for (int j = ddims; j > 2; j--) {
                    int ll = ldims - j;
                    int rr = rdims - j;
                    int dd = ddims - j;

                    int lsize = ll >= 0 ? left.shape[ll] : 1;
                    int rsize = rr >= 0 ? right.shape[rr] : 1;
                    int dsize = output.shape[dd];

                    stride /= dsize;
                    int ind = remaining / stride; // value at this index
                    remaining %= stride;

                    if (ind == 0) {
                        continue;
                    }

                    // if broadcast (ie. shape is 1), do not update ind
                    if (lsize != 1) {
                        lind += ind * stride;
                    }
                    if (rsize != 1) {
                        rind += ind * stride;
                    }
                }


                // get offsets
                int loffset = lx * ly * lind;
                int roffset = rx * ry * rind;
                int doffset = lx * ry * i;

                int di = doffset;
                for (int x = 0; x < lx; x++) {
                    for (int y = 0; y < ry; y++) {
                        float sum = 0;
                        int ri = y + roffset;
                        int li = x * ly + loffset;
                        for (int j = 0; j < ly; j++) {
                            //sum += l[x, i] * r[i, y];
                            var a = lv[li];
                            var b = rv[ri];
                            sum += a * b;
                            ri += ry;
                            li++;
                        }
                        //dest[x, y] += sum;
                        dv[di] = sum;
                        di++;
                    }
                }
            }
        }

    }
}