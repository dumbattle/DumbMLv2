using Unity.Jobs;
using Unity.Collections;


namespace DumbML.BLAS.CPU {
    public struct BroadcastJob : IJobParallelFor {
        [ReadOnly]
        NativeArray<float> src;
        NativeArray<float> result;

        [ReadOnly]
        NativeArray<int> srcShape;
        [ReadOnly]
        NativeArray<int> shape;

        int isize;
        int dsize;
        int shapeRank;
        int srcRank;

        public BroadcastJob(FloatCPUTensorBuffer input, int[] shape, FloatCPUTensorBuffer dest) {
            src = input.buffer;
            result = dest.buffer;

            srcShape = new NativeArray<int>(input.shape, Allocator.TempJob);
            this.shape = new NativeArray<int>(shape, Allocator.TempJob);

            isize = input.size;
            dsize = dest.size;
            shapeRank = dest.Rank();
            srcRank = input.Rank(); ;
        }
        public void Dispose() {
            srcShape.Dispose();
            shape.Dispose();
        }
        public void Execute(int index) {
            int offset = 0;
            int srcStride = isize;
            int dStride = dsize;
            int remaining = index;


            for (int i = shapeRank; i > 0; i--) {
                int srcind = srcRank - i;
                int shpind = shapeRank - i;

                bool isBroadcasted = srcind < 0 || srcShape[srcind] != shape[shpind];

                int dimSize = shape[shpind];
                dStride /= dimSize;

                int indCount = remaining / dStride;
                remaining = remaining % dStride;

                if (!isBroadcasted) {
                    srcStride /= dimSize;
                    offset += indCount * srcStride;
                }
            }

            result[index] = src[offset];
        }
    }

}