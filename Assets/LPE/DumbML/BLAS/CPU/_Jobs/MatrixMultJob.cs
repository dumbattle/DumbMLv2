using Unity.Jobs;
using Unity.Collections;

namespace DumbML.BLAS.CPU {
    public struct MatrixMultJob : IJobParallelFor {
        public NativeArray<float> result;
     
        [ReadOnly]
        NativeArray<float> left;
        [ReadOnly]
        NativeArray<float> right;

        [ReadOnly]
        NativeArray<int> lshape;
        [ReadOnly]
        NativeArray<int> rshape;
        [ReadOnly]
        NativeArray<int> dshape;

        int lrank;
        int rrank;
        int drank;

        int batchCountL;
        int batchCountR;
        int count;

        bool transposeL;
        bool transposeR;

        public MatrixMultJob(FloatCPUTensorBuffer l, FloatCPUTensorBuffer r, FloatCPUTensorBuffer dest,
                                   bool transposeL, bool transposeR, int bcl, int bcr) {
            left = l.buffer;
            right = r.buffer;
            result = dest.buffer;

            lshape = new NativeArray<int>(l.shape, Allocator.TempJob);
            rshape = new NativeArray<int>(r.shape, Allocator.TempJob);
            dshape = new NativeArray<int>(dest.shape, Allocator.TempJob);

            lrank = l.Rank();
            rrank = r.Rank();
            drank = dest.Rank();

            batchCountL = bcl;
            batchCountR = bcr;

            count = dest.size;
            this.transposeL = transposeL;
            this.transposeR = transposeR;
        }

        public void Execute(int index) {
            int matSize = dshape[drank - 1] * dshape[drank - 2];

            int batch = index / matSize; // which batch are we on
            int batch_ind = index % matSize; // index inside that batch

            int ldims = lrank;
            int rdims = rrank;
            int ddims = drank;

            int lx = lshape[transposeL ? ldims - 1 : ldims - 2];
            int ly = lshape[transposeL ? ldims - 2 : ldims - 1];
            int rx = rshape[transposeR ? rdims - 1 : rdims - 2];
            int ry = rshape[transposeR ? rdims - 2 : rdims - 1];

            // get left right batches
            int lind = 0;
            int rind = 0;
            int remaining = batch;
            int stride = count / matSize;
            int strideL = batchCountL;
            int strideR = batchCountR;

            for (int j = ddims; j > 2; j--) {
                int ll = ldims - j;
                int rr = rdims - j;
                int dd = ddims - j;

                int lsize = ll >= 0 ? lshape[ll] : 1;
                int rsize = rr >= 0 ? rshape[rr] : 1;
                int dsize = dshape[dd];

                stride /= dsize;
                strideL /= lsize;
                strideR /= rsize;
                int ind = remaining / stride; // value at this index
                remaining %= stride;


                // if broadcast (ie. shape is 1), do not update ind
                if (lsize != 1) {
                    lind += ind * strideL;
                }
                if (rsize != 1) {
                    rind += ind * strideR;
                }
            }

            // compute 
            int loffset = lx * ly * lind;
            int roffset = rx * ry * rind;
            int doffset = lx * ry * batch;

            int lxStride = transposeL ? 1 : ly;
            int liStride = transposeL ? lx : 1;

            int riStride = transposeR ? 1 : ry;
            int ryStride = transposeR ? rx : 1;


            int x = batch_ind / ry;
            int y = batch_ind % ry;

            float sum = 0;
            int ri = y * ryStride + roffset;
            int li = x * lxStride + loffset;

            for (int k = 0; k < ly; k++) {
                float a = left[li];
                float b = right[ri];
                sum += a * b;
                ri += riStride;
                li += liStride;
            }
            result[index] = sum;
        }

        public void Dispose() {
            lshape.Dispose();
            rshape.Dispose();
            dshape.Dispose();
        }
    }
}