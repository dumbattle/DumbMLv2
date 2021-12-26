using System;
using System.Collections.Generic;
using LPE;


namespace DumbML.BLAS.CPU {
    public static class Transpose {
        static ComputeDelegate cd = new ComputeDelegate();

        public static void Compute(FloatCPUTensorBuffer src, int[] perm, FloatCPUTensorBuffer dest) {
            // TODO - check shape

            Compute(src.buffer, src.shape, perm, dest.buffer);
        }
        public static void Compute(float[] src, int[] shape, int[] perm, float[] dest) {
            var shapeList = Utils.GetIntList();
            shapeList.Clear();
            shapeList.AddRange(shape);
            Compute(src, shapeList, perm, dest);
            Utils.Return(shapeList);
        }
        public static void Compute(float[] src, List<int> shape, int[] perm, float[] dest) {
            int inputSize = 1;
            foreach (var i in shape) {
                inputSize *= i;
            }

            if (inputSize < src.Length) {
                throw new ArgumentException("Destination buffer is not large enough");
            }

            int minThreadSize = 100;

            int threadCount = inputSize / minThreadSize;
            int[] strides = Utils.GetIntArr();
            GetStrides(shape, strides);

            if (threadCount <= 1) {
                cd.Init(src, shape, perm, dest, 1, strides);
                cd.CallForwardAction(0);
            }
            else {
                cd.Init(src, shape, perm, dest, 1, strides);
                var cb = ThreadPool.ForWithCallback(0, threadCount, cd.CallForwardAction);
                cb.Wait();
            }
            Utils.Return(strides);
        }

        static void GetStrides(List<int> shape, int[] result) {
            int stride = 1;

            for (int i = shape.Count - 1; i >= 0; i--) {
                result[i] = stride;

                int dimSize = shape[i];
                stride *= dimSize;
            }
        }

        class ComputeDelegate {
            public Action<int> CallForwardAction;

            float[] src;
            List<int> shape;
            int size;
            int[] perm;
            float[] dest;
            int threadCount;
            int[] strides;

            public ComputeDelegate() {
                CallForwardAction = CallForward;
            }

            public void Init(float[] src, List<int> shape, int[] perm, float[] dest, int threadCount, int[] strides) {
                this.src = src;
                this.shape = shape;
                this.perm = perm;
                this.dest = dest;
                this.threadCount = threadCount;
                this.strides = strides;

                size = 1;
                foreach (var s in shape) {
                    size *= s;
                }
            }

            void CallForward(int t) {
                int start = size * t / threadCount;
                int end = size * (t + 1) / threadCount;

                for (int i = start; i < end; i++) {
                    int stride = size;
                    int remaining = i;
                    int offset = 0;

                    for (int axis = 0; axis < shape.Count; axis++) {
                        int dimSize = shape[perm[axis]];
                        stride /= dimSize;

                        int indCount = remaining / stride;
                        remaining = remaining % stride;

                        offset += indCount * strides[perm[axis]];
                    }

                    dest[i] = src[offset];
                }
            }
        }
    }

    public static class Broadcast {
        public static void Compute(FloatCPUTensorBuffer input, int[] shape, FloatCPUTensorBuffer dest) {
      
        }
    }

}