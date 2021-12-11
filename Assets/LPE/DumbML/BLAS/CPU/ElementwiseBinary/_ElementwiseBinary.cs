using System;
using LPE;

namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        static class Computation<T> where T : ComputeDelegateCache, new() {
            static ObjectPool<T> cachePool
                = new ObjectPool<T>(() => new T());

            public static void Forward(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, FloatCPUTensorBuffer output) {
                T compute = cachePool.Get();

                PartitionInfo pi = CheckShape(a.shape, b.shape, output.shape);
          
                compute.SetForward(a, b, output, pi);
                var cb = ThreadPool.ForWithCallback(0, pi.batchCount, compute.CallForwardAction);
                cb.Wait();

                //for (int i = 0; i < pi.batchCount; i++) {
                //    compute.CallForwardAction(i);
                //}
                cachePool.Return(compute);
            }

            static PartitionInfo CheckShape(int[] l, int[] r, int[] d) {
                int threadSize = 10; // min array size each thread computes (don't want too many small threads) (arbitrarily set)


                int ldims = l.Length;
                int rdims = r.Length;
                int ddims = UnityEngine.Mathf.Max(ldims, rdims);

                if (ddims != d.Length) {
                    throw new InvalidOperationException($"Output Tensors do not have correcct rank\n  Expected{ddims}\n  Got:{d.ContentString()}");
                }

                int strideSize = 1;
                int batchCount = 1;
                int batchCountL = 1;
                int batchCountR = 1;
                bool strideDone = false;

                for (int i = 1; i <= ddims; i++) {
                    int dimSize = -1;

                    int li = ldims - i;
                    int ri = rdims - i;
                    int di = ddims - i;

                    int lsize = li >= 0 ? l[li] : 1;
                    int rsize = ri >= 0 ? r[ri] : 1;

                    // same
                    if (rsize == lsize) {
                        dimSize = rsize;
                    }
                    // left is broadcastable to right
                    else if (lsize == 1) {
                        dimSize = rsize;
                        strideDone = true;
                    }
                    // right is broadcastable to left
                    else if (rsize == 1) {
                        dimSize = lsize;
                        strideDone = true;
                    }

                    // not compatable
                    if (dimSize == -1) {
                        throw new InvalidOperationException(
                            $"Input Tensors do not have compatible dimensions: {l.ContentString()}, {r.ContentString()}"
                        );
                    }

                    // dest doesnt have correct shape
                    if (dimSize != d[di]) {
                        throw new InvalidOperationException(
                            $"Destination tensor does not have compatable dimensions: {d.ContentString()} Expected '{dimSize}' at index '{di}'"
                        );
                    }

                    if (!strideDone && strideSize < threadSize ) {
                        strideSize *= dimSize;
                        strideDone = true;
                    }
                    else {
                        batchCount *= dimSize;
                        batchCountL *= lsize;
                        batchCountR *= rsize;
                    }
                }

                return new PartitionInfo() {
                    batchCount = batchCount,
                    lBatchCount = batchCountL,
                    rBatchCount = batchCountR,
                    stride = strideSize,
                };
            }

         
        }


        abstract class ComputeDelegateCache {
            FloatCPUTensorBuffer left;
            FloatCPUTensorBuffer right;
            FloatCPUTensorBuffer output;

            PartitionInfo pi;

            public Action<int> CallForwardAction;

            public ComputeDelegateCache() {
                CallForwardAction = CallForward;
            }


            public void SetForward(FloatCPUTensorBuffer left, FloatCPUTensorBuffer right, FloatCPUTensorBuffer output, PartitionInfo pi) {
                this.left = left;
                this.right = right;
                this.output = output;
                this.pi = pi;
            }

            void CallForward(int i) {
                int ldims = left.Rank();
                int rdims = right.Rank();
                int ddims = output.Rank();

                float[] lv = left.buffer;
                float[] rv = right.buffer;
                float[] ov = output.buffer;

                int lind = 0;
                int rind = 0;
                int remaining = i;
                int stride = pi.batchCount;
                int strideL = pi.lBatchCount;
                int strideR = pi.rBatchCount;

                for (int j = ddims; j > 0; j--) {
                    int ll = ldims - j;
                    int rr = rdims - j;
                    int dd = ddims - j;

                    int lsize = ll >= 0 ? left.shape[ll] : 1;
                    int rsize = rr >= 0 ? right.shape[rr] : 1;
                    int dsize = output.shape[dd];

                    stride /= dsize;
                    strideL /= lsize;
                    strideR /= rsize;
                    if (stride == 0) {
                        break;
                    }
                    int ind = remaining / stride; // value at this index
                    remaining %= stride;

                    if (ind == 0) {
                        continue;
                    }

                    // if broadcast (ie. shape is 1), do not update ind
                    if (lsize != 1) {
                        lind += ind * strideL;
                    }
                    if (rsize != 1) {
                        rind += ind * strideR;
                    }
                }

                int loffset = pi.stride * lind;
                int roffset = pi.stride * rind;
                int doffset = pi.stride * i;

                Forward(lv, rv, ov, loffset, roffset, doffset, pi.stride);
            }


            public abstract void Forward(float[] l, float[] r, float[] d, int startL, int startR, int startD, int stride);
        }
        struct PartitionInfo {
            public int batchCount;
            public int lBatchCount;
            public int rBatchCount;
            public int stride;
        }
    }
}
