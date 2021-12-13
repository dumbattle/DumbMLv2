using LPE;
using System.Threading;
using System.Collections.Generic;


namespace DumbML.BLAS.CPU {
    public static partial class Reduction {

    
        static class Reduce<T> where T: Reducer, new() {
            static ObjectPool<ReduceDelegate<T>> pool = new ObjectPool<ReduceDelegate<T>>(() => new ReduceDelegate<T>());

            public static void Compute(FloatCPUTensorBuffer src, int[] axis, FloatCPUTensorBuffer dest) {
                SetValues.Zero(dest);


                // TODO check shapes

                var forward = pool.Get();
                forward.src = src;
                forward.axis = axis;
                forward.dest = dest;
                forward.SetSize(dest.size);
                var cb = LPE.ThreadPool.ForWithCallback(0, dest.size, forward.ForwardAction);
                cb.Wait();

                pool.Return(forward);
            }
        
        }

        class ReduceDelegate<T> where T : Reducer, new() {
            List<T> reducers = new List<T>();

            public FloatCPUTensorBuffer src;
            public int[] axis;
            public FloatCPUTensorBuffer dest;

            public System.Action<int> ForwardAction;

            public ReduceDelegate() {
                ForwardAction = Run;
            }

            public void SetSize(int size) {
                while (reducers.Count < size) {
                    reducers.Add(new T());
                }
            }
            public void Run(int o) {
                T reducer = reducers[o];
                int reductionSize = src.size / dest.size; // number of input elements per output element
                // get starting index
                int ind = o;
                int istride = src.size;
                int dstride = dest.size;
                int start = 0;
                // TODO if keep axis, use loop 'a' as instead
                int daxis = 0;

                for (int a = 0; a < src.shape.Length; a++) {
                    int dimSize = src.shape[a];
                    istride /= dimSize;

                    if (Contains(axis, a)) {
                        continue;
                    }
                    dstride /= dest.shape[daxis];

                    int dimCount = ind / dstride;
                    int remaining = ind % dstride;


                    start += istride * dimCount;

                    ind = remaining;
                    daxis++;
                }

                reducer.Reset();
                for (int i = 0; i < reductionSize; i++) {
                    int rStride = src.size / dest.size;
                    int srcStride = src.size;
                    int offset = 0;
                    ind = i;

                    for (int a = 0; a < src.shape.Length; a++) {
                        int dimSize = src.shape[a];
                        srcStride /= dimSize;

                        if (!Contains(axis, a)) {
                            continue;
                        }

                        rStride /= dimSize;

                        int dimCount = ind / rStride;
                        ind = ind % rStride;

                        offset += dimCount * srcStride;
                    }
                    reducer.Update(src.buffer[start + offset]);
                }

                dest.buffer[o] = reducer.Complete();

            }

            static bool Contains(int[] arr, int val) {
                if (arr == null) {
                    return true;
                }
                foreach (var a in arr) {
                    if (a == val) {
                        return true;
                    }
                }
                return false;
            }


        }
        abstract class Reducer {

            public abstract void Reset();
            public abstract void Update(float val);
            public abstract float Complete();
        }
    }
}