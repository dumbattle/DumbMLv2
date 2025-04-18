﻿using Unity.Jobs;
using Unity.Collections;


namespace DumbML.BLAS.CPU {
    public static class ReductionJob {
        public interface IImplementation {
            void Start();
            void Next(float v);
            float Complete();
        }

        public struct Job<T> : IJobParallelFor where T : struct, IImplementation {
            public NativeArray<float> result;

            [ReadOnly]
            NativeArray<float> src;

            [ReadOnly]
            NativeArray<int> axis;
            [ReadOnly]
            NativeArray<int> srcShape;
            [ReadOnly]
            NativeArray<int> destShape;

            int axisLength;
            int srcRank;
            int srcSize;
            int destSize;

            public Job(FloatCPUTensorBuffer src, int[] axis, FloatCPUTensorBuffer dest) {
                this.src = src.buffer;
                result = dest.buffer;

                srcShape = new NativeArray<int>(src.shape, Allocator.TempJob);
                destShape = new NativeArray<int>(dest.shape, Allocator.TempJob);

                if (axis != null) {
                    this.axis = new NativeArray<int>(axis, Allocator.TempJob);
                    axisLength = axis.Length;
                }
                else {
                    this.axis = new NativeArray<int>(0, Allocator.TempJob);
                    axisLength = -1;
                }
                srcRank = src.Rank();
                srcSize = src.size;
                destSize = dest.size;
            }

            public void Execute(int index) {
                T reducer = default;
                reducer.Start();
                int reductionSize = srcSize / destSize; // number of input elements per output element
                // get starting index
                int ind = index;
                int istride = srcSize;
                int dstride = destSize;
                int start = 0;
                // TODO if keep axis, use loop 'a' as instead
                int daxis = 0;

                for (int a = 0; a < srcRank; a++) {
                    int dimSize = srcShape[a];
                    istride /= dimSize;

                    if (AxisContain(a)) {
                        continue;
                    }
                    dstride /= destShape[daxis];

                    int dimCount = ind / dstride;
                    int remaining = ind % dstride;


                    start += istride * dimCount;

                    ind = remaining;
                    daxis++;
                }

                for (int i = 0; i < reductionSize; i++) {
                    int rStride = reductionSize;
                    int srcStride = srcSize;
                    int offset = 0;
                    ind = i;

                    for (int a = 0; a < srcRank; a++) {
                        int dimSize = srcShape[a];
                        srcStride /= dimSize;

                        if (!AxisContain(a)) {
                            continue;
                        }

                        rStride /= dimSize;

                        int dimCount = ind / rStride;
                        ind = ind % rStride;

                        offset += dimCount * srcStride;
                    }
                    reducer.Next(src[start + offset]);
                }

                result[index] = reducer.Complete();
            }

            public void Dispose() {
                axis.Dispose();
                srcShape.Dispose();
                destShape.Dispose();
            }

            bool AxisContain(int a) {
                if (axisLength == 0) {
                    return true;
                }
                if (axisLength == -1) {
                    return true;
                }
                for (int i = 0; i < axisLength; i++) {
                    var val = axis[i];
                    if (a == val) {
                        return true;
                    }
                }
                return false;
            }


        }
    }


}
