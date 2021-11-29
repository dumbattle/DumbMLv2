using System;
using LPE;

namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        static class Computation<T> where T : ComputeDelegateCache, new() {
            static ObjectPool<T> cachePool
                = new ObjectPool<T>(() => new T());

            public static void Forward(CPUTensorBuffer a, CPUTensorBuffer b, CPUTensorBuffer output) {
                int dims = a.Rank();

                // check dims all match
                if (b.Rank() != dims) {
                    throw new InvalidOperationException($"Tensors do not have compatible dimensions: {a.shape.ContentString()}, {b.shape.ContentString()}");
                }
                if (output.Rank() != dims) {
                    throw new InvalidOperationException($"Desination Tensor does not have correct rank: {dims} vs {output.Rank()}");
                }

                T compute = cachePool.Get();


                // 1D - No parallel
                if (dims == 1) {
                    compute.Forward(a.buffer, b.buffer, output.buffer, 0, output.size);
                }
                else {
                    // parallel
                    compute.SetForward(a, b, output, a.size / a.shape[0]);
                    var cb = ThreadPool.ForWithCallback(0, a.shape[0], compute.CallForwardAction);
                    cb.Wait();

                    cachePool.Return(compute);
                }
            }
        }


        abstract class ComputeDelegateCache {
            CPUTensorBuffer left;
            CPUTensorBuffer right;
            CPUTensorBuffer output;

            int stride;

            public Action<int> CallForwardAction;

            public ComputeDelegateCache() {
                CallForwardAction = CallForward;
            }


            public void SetForward(CPUTensorBuffer left, CPUTensorBuffer right, CPUTensorBuffer output, int stride) {
                this.left = left;
                this.right = right;
                this.output = output;
                this.stride = stride;
            }

            void CallForward(int i) {
                float[] lv = left.buffer;
                float[] rv = right.buffer;
                float[] ov = output.buffer;

                int start = stride * i;
                int end = stride * (i + 1);
                Forward(lv, rv, ov, start, end);
            }


            public abstract void Forward(float[] l, float[] r, float[] d, int start, int end);
        }
    }
}
