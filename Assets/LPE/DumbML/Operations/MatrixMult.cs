using System;
using System.Collections.Generic;
using UnityEngine;

namespace DumbML {
    public class MatrixMult : Operation {
        bool transposeL;
        bool transposeR;
        int[] shapeActual;

        public MatrixMult(Operation l, Operation r, bool transposeL = false, bool transposeR = false) {
            int aRank = l.shape.Length;
            int bRank = r.shape.Length;
            this.transposeL = transposeL;
            this.transposeR = transposeR;

            // Compatible matrix shapes
            if (l.shape[transposeL ? aRank - 2 : aRank - 1] != r.shape[transposeR ? bRank - 1 : bRank - 2]) {
                throw new ArgumentException($"Cannot MatrixMult tensors of shapes: {l.shape.ContentString()} by {r.shape.ContentString()}");
            }


            shapeActual = GetShape(l.shape, r.shape, shapeActual, transposeL, transposeR);
            if (shapeActual == null) {
                Debug.Log("???????????????????????????");
            }
            BuildOp(shapeActual, DType.Float, l, r);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            shapeActual = GetShape(inputs[0].shape, inputs[1].shape, shapeActual, transposeL, transposeR);
            result.SetShape(shapeActual);
            BLAS.Engine.Compute.MatrixMult(inputs[0], inputs[1], result, transposeL, transposeR);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            Operation agrad = null;
            Operation bgrad = null;

            if (!transposeL && !transposeR) {
                agrad =  new MatrixMult(error, inputs[1], false, true);
                bgrad = new MatrixMult(inputs[0], error, true, false);
            }
            else if (!transposeL && transposeR) {
                agrad = new MatrixMult(error, inputs[1], false, false);
                bgrad = new MatrixMult(error, inputs[0], true, false);
            }
            else if (transposeL && !transposeR) {
                agrad = new MatrixMult(inputs[1], error, false, true);
                bgrad = new MatrixMult(inputs[0], error, false, false);
            }
            else if (transposeL && transposeR) {
                agrad = new MatrixMult(inputs[1], error, true, true);
                bgrad = new MatrixMult(error, inputs[0], true, true);
            }
            var (a, b) = GetGradientReduction(inputs[0].shape, inputs[1].shape, error.shape);

            return new Operation[] { 
                a.Length > 0 ? new Reshape(new ReduceSum(agrad, inputs[0]), inputs[0]) : agrad,
                b.Length > 0 ? new Reshape(new ReduceSum(bgrad, inputs[1]), inputs[1]) : bgrad
            };
        }
        static (int[], int[]) GetGradientReduction(int[] ashape, int[] bshape, int[] eshape) {
            List<int> a = new List<int>();
            List<int> b = new List<int>();



            for (int i = eshape.Length; i > 2; i--) {
                int adim = ashape.Length - i;
                int bdim = bshape.Length - i;
                int edim = eshape.Length - i;

                if (adim < 0) {
                    a.Add(edim);
                }
                else if (bdim < 0) {
                    b.Add(edim);
                }
                else if (eshape[edim] != 1) {
                    if (ashape[adim] == 1) {
                        a.Add(edim);
                    }
                    else if (bshape[bdim] == 1) {
                        b.Add(edim);
                    }
                }
            }


            return (a.ToArray(), b.ToArray());
        }
        static int[] GetShape(int[] left, int[] right, int[] result = null, bool transposeL = false, bool transposeR = false) {
            int ldims = left.Length;
            int rdims = right.Length;
            int ddims = Mathf.Max(ldims, rdims);

             result = result ?? new int[ddims];
            // check ranks > 2
            if (ldims < 2) {
                throw new ArgumentException($"MatrixMult requires tensors to have dimension of at least 2. Got shape: {left.ContentString()}");
            }
            if (rdims < 2) {
                throw new ArgumentException($"MatrixMult requires tensors to have dimension of at least 2. Got shape: {right.ContentString()}");
            }
            if (result.Length != ddims) {
                throw new ArgumentException($"MM");
            }


            // check leading dimensions
            // can't start from 0 because l and r might have different ranks (ie. 1 of them might have implicit leading dimensions)
            // instead we use distancce from end to get dimension
            // negative = implied dimension of [1]
            // stop at 2 because we 2 dimensions are for matmult
            for (int i = ddims; i > 2; i--) {
                int dimSize = -1;

                int li = ldims - i;
                int ri = rdims - i;
                int di = ddims - i;

                int lsize = li >= 0 ? left[li] : 1;
                int rsize = ri >= 0 ? right[ri] : 1;

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
                else {
                    throw new InvalidOperationException($"Tensors do not have compatible dimensions: {left.ContentString()}, {right.ContentString()}");

                }
                result[di] = dimSize;
            }

            // check shape compatability


            int lx = left[ldims - (transposeL ? 1 : 2)];
            int ly = left[ldims - (transposeL ? 2 : 1)];
            int rx = right[rdims - (transposeR ? 1 : 2)];
            int ry = right[rdims - (transposeR ? 2 : 1)];

            if (ly != rx) {
                throw new InvalidOperationException($"Tensors do not have compatible dimensions: {left.ContentString()}, {right.ContentString()}");
            }
            result[ddims - 2] = lx;
            result[ddims - 1] = ry;
            return result;
        }
    }
}
