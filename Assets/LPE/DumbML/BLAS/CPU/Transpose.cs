using System;
using LPE;


namespace DumbML.BLAS.CPU {
    public static class Transpose {
        static ComputeDelegate cd = new ComputeDelegate();

        public static void Compute(FloatCPUTensorBuffer src, int[] perm, FloatCPUTensorBuffer dest) {
            if (!ShapeUtility.SameShape(src.shape, dest.shape)) {
                throw new ArgumentException("Buffers are not the same shape");
            }

            Compute(src, perm, dest);
        }
        public static void Compute_IgnoreShape(FloatCPUTensorBuffer src, int[] perm, FloatCPUTensorBuffer dest) {
            if (dest.capacity < src.size) {
                throw new ArgumentException("Destination buffer is not large enough");
            }

            int minThreadSize = 100;

            int threadCount = src.size / minThreadSize;
            if (threadCount <= 1) {
                cd.Init(src, perm, dest, 1, GetStrides(src.shape));
                cd.CallForwardAction(0);
            }
            else {
                cd.Init(src, perm, dest, 1, GetStrides(src.shape));
                var cb = ThreadPool.ForWithCallback(0, threadCount, cd.CallForwardAction);
                cb.Wait();
            }
        }

        static int[] GetStrides(int[] shape) {
            int[] result = Utils.intArr;

            int stride = 1;

            for (int i = shape.Length - 1; i >= 0; i--) {
                result[i] = stride;

                int dimSize = shape[i];
                stride *= dimSize;
            }

            return result;
        }

        class ComputeDelegate {
            public Action<int> CallForwardAction;

            FloatCPUTensorBuffer src;
            int[] perm; 
            FloatCPUTensorBuffer dest;
            int threadCount;
            int[] strides;

            public ComputeDelegate() {
                CallForwardAction = CallForward;
            }

            public void Init(FloatCPUTensorBuffer src, int[] perm, FloatCPUTensorBuffer dest, int threadCount, int[] strides) {
                this.src = src;
                this.perm = perm;
                this.dest = dest;
                this.threadCount = threadCount;
                this.strides = strides;
            }

            void CallForward(int t) {
                int start = src.size * t / threadCount;
                int end = src.size * (t + 1) / threadCount;

                for (int i = start; i < end; i++) {
                    int stride = src.size;
                    int remaining = i;
                    int offset = 0;

                    for (int axis = 0; axis < perm.Length; axis++) {
                        int dimSize = src.shape[perm[axis]];
                        stride /= dimSize;

                        int indCount = remaining / stride;
                        remaining = remaining % stride;

                        offset += indCount * strides[perm[axis]];
                    }

                    dest.buffer[i] = src.buffer[offset];
                }
            }
        }

    }
}