using Unity.Jobs;
using Unity.Collections;


namespace DumbML.BLAS.CPU {
    public static class ElementwiseBinaryJob {
        
        public interface IImplementation {
            float Forward(float l, float r);
        }

        public struct Job<T> : IJobParallelFor where T : struct, IImplementation {
            [ReadOnly]
            public NativeArray<float> lv;
            [ReadOnly]
            public NativeArray<float> rv;
            public NativeArray<float> ov;

            [ReadOnly]
            public NativeArray<int> lShape;
            [ReadOnly]
            public NativeArray<int> rShape;
            [ReadOnly]
            public NativeArray<int> oShape;

            public int ldims;
            public int rdims;
            public int odims;

            public int lsize;
            public int rsize;
            public int osize;

            public void Init(FloatCPUTensorBuffer left, FloatCPUTensorBuffer right, FloatCPUTensorBuffer output) {
                lv = left.buffer;
                rv = right.buffer;
                ov = output.buffer;

                lShape = new NativeArray<int>(left.shape, Allocator.TempJob);
                rShape = new NativeArray<int>(right.shape, Allocator.TempJob);
                oShape = new NativeArray<int>(output.shape, Allocator.TempJob);

                ldims = left.Rank();
                rdims = right.Rank();
                odims = output.Rank();

                lsize = left.size;
                rsize = right.size;
                osize = output.size;
            }
            
            public void Dispose() {
                lShape.Dispose();
                rShape.Dispose();
                oShape.Dispose();
            }
            
            public void Execute(int i) {
                int lind = 0;
                int rind = 0;
                int remaining = i;

                int strideO = osize;
                int strideL = lsize;
                int strideR = rsize;

                for (int j = odims; j > 0; j--) {
                    int ll = ldims - j;
                    int rr = rdims - j;
                    int dd = odims - j;

                    int ldsize = ll >= 0 ? lShape[ll] : 1;
                    int rdsize = rr >= 0 ? rShape[rr] : 1;
                    int odsize = oShape[dd];

                    strideO /= odsize;
                    strideL /= ldsize;
                    strideR /= rdsize;

                    if (strideO == 0) {
                        break;
                    }
                    int ind = remaining / strideO; // value at this index
                    remaining %= strideO;

                    if (ind == 0) {
                        continue;
                    }

                    // if broadcast (ie. shape is 1), do not update ind
                    if (ldsize != 1) {
                        lind += ind * strideL;
                    }
                    if (rdsize != 1) {
                        rind += ind * strideR;
                    }
                }

                T t = default;
                ov[i] =  t.Forward(lv[lind], rv[rind]);
            }
        }
    
    
    }
}
