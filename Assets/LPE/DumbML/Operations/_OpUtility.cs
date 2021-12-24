using System;
using System.Collections.Generic;


namespace DumbML {
    public static class OpUtility {
        public static int[] GetBroadcastShape(int[] a, int[] b, int[] result) {
            int adims = a.Length;
            int bdims = b.Length;
            int rdims = UnityEngine.Mathf.Max(adims, bdims);
            result = result ?? new int[rdims];

            if (result.Length != rdims) {
                throw new ArgumentException($"incorrect result length\nExpected: {rdims}\nGot: {result.Length}");
            }

            for (int i = 1; i <= rdims; i++) {
                int dimSize;
                int ai = adims - i;
                int bi = bdims - i;
                int ri = rdims - i;

                int asize = ai >= 0 ? a[ai] : 1;
                int bsize = bi >= 0 ? b[bi] : 1;

                // same
                if (asize == bsize) {
                    dimSize = asize;
                }
                // left is broadcastable to right
                else if (asize == 1) {
                    dimSize = bsize;
                }
                // right is broadcastable to left
                else if (bsize == 1) {
                    dimSize = asize;
                }
                // not compatable
                else {
                    throw new InvalidOperationException(
                        $"Shapes do not have broadcastable dimensions: {a.ContentString()}, {b.ContentString()}"
                    );
                }

                result[ri] = dimSize;
            }

            return result;
        }
    
        public static List<int> BroadcastBackwardsReductionShape(int[] inputShape, int[] errShape) {
            List<int> result = new List<int>();

            for (int i = errShape.Length; i > 0; i--) {
                int idim = inputShape.Length - i;
                int edim = errShape.Length - i;

                if (idim < 0) {
                    result.Add(edim);
                }
                else if (errShape[edim] != 1 && inputShape[idim] == 1) {
                    result.Add(edim);
                }
            }
            return result;
        }
    }

    //public class Reshape : Operation {
    //    public Reshape(Operation op, params int[] shape) {
    //        BuildOp(shape, op.dtype, op);
    //    }


    //    public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
    //    }
    //    public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
    //    }
    //}
}
