using System;
using LPE;
using Unity.Jobs;


namespace DumbML.BLAS.CPU {
    public static partial class ElementwiseBinary {
        static class Computation<T> where T : struct, ElementwiseBinaryJob.IImplementation {
            public static void Forward(FloatCPUTensorBuffer a, FloatCPUTensorBuffer b, FloatCPUTensorBuffer output, ElementwiseBinaryJob.Job<T> j) {
                CheckShape(a.shape, b.shape, output.shape);
                j.Init(a, b, output);
                var h = j.Schedule(output.size, 1);
                h.Complete();
                j.Dispose();
            }

            static void CheckShape(int[] l, int[] r, int[] d) {
                int ldims = l.Length;
                int rdims = r.Length;
                int ddims = UnityEngine.Mathf.Max(ldims, rdims);

                if (ddims != d.Length) {
                    throw new InvalidOperationException($"Output Tensors do not have correcct rank\n  Expected{ddims}\n  Got:{d.ContentString()}");
                }

    
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
                    }
                    // right is broadcastable to left
                    else if (rsize == 1) {
                        dimSize = lsize;
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
                }
            }
        }

    }
}
