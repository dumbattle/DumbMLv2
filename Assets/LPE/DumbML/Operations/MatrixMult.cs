using System;
using UnityEngine;

namespace DumbML {
    public class MatrixMult : Operation {
        bool transposeL;
        bool transposeR;
        public MatrixMult(Operation l, Operation r, bool transposeL = false, bool transposeR = false) {
            int aRank = l.shape.Length;
            int bRank = r.shape.Length;
            this.transposeL = transposeL;
            this.transposeR = transposeR;

            // Compatible matrix shapes
            if (l.shape[transposeL ? aRank - 2 : aRank - 1] != r.shape[transposeR ? bRank - 1 : bRank - 2]) {
                throw new System.ArgumentException($"Cannot MatrixMult tensors of shapes: {l.shape.ContentString()} by {r.shape.ContentString()}");
            }


            int[] shape = GetShape(l.shape, r.shape, transposeL, transposeR);

            BuildOp(shape, DType.Float, l, r);
        }


        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            BLAS.Engine.Compute.MatrixMult(inputs[0], inputs[1], result, transposeL, transposeR);
        }


        public override void Backward(ITensorBuffer[] inputs, ITensorBuffer output, ITensorBuffer error, ITensorBuffer[] results) {
            if (!transposeL && !transposeR) {
                BLAS.Engine.Compute.MatrixMult(error, inputs[1], results[0], false, true);
                BLAS.Engine.Compute.MatrixMult(inputs[0], error, results[1], true, false);
            }
            else if (!transposeL && transposeR) {
                BLAS.Engine.Compute.MatrixMult(error, inputs[1], results[0], false, false);
                BLAS.Engine.Compute.MatrixMult(error, inputs[0], results[1], true, false);
            }
            else if (transposeL && !transposeR) {
                BLAS.Engine.Compute.MatrixMult(inputs[1], error, results[0], false, true);
                BLAS.Engine.Compute.MatrixMult(inputs[0], error, results[1], false, false);
            }
            else if (transposeL && transposeR) {
                BLAS.Engine.Compute.MatrixMult(inputs[1], error, results[0], true, true);
                BLAS.Engine.Compute.MatrixMult(error, inputs[0], results[1], true, true);
            }
            /*
             tensorflow implementation
              t_a = op.get_attr("transpose_a")
              t_b = op.get_attr("transpose_b")
              a = math_ops.conj(op.inputs[0])
              b = math_ops.conj(op.inputs[1])
              if not t_a and not t_b:
                grad_a = gen_math_ops.mat_mul(grad, b, transpose_b=True)
                grad_b = gen_math_ops.mat_mul(a, grad, transpose_a=True)
              elif not t_a and t_b:
                grad_a = gen_math_ops.mat_mul(grad, b)
                grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True)
              elif t_a and not t_b:
                grad_a = gen_math_ops.mat_mul(b, grad, transpose_b=True)
                grad_b = gen_math_ops.mat_mul(a, grad)
              elif t_a and t_b:
                grad_a = gen_math_ops.mat_mul(b, grad, transpose_a=True, transpose_b=True)
                grad_b = gen_math_ops.mat_mul(grad, a, transpose_a=True, transpose_b=True)
            */
        }


        static int[] GetShape(int[] left, int[] right, bool transposeL = false, bool transposeR = false) {
            int ldims = left.Length;
            int rdims = right.Length;
            int ddims = Mathf.Max(ldims, rdims);

            int[] result = new int[ddims];
            // check ranks > 2
            if (ldims < 2) {
                throw new ArgumentException($"MatrixMult requires tensors to have dimension of at least 2. Got shape: {left.ContentString()}");
            }
            if (rdims < 2) {
                throw new ArgumentException($"MatrixMult requires tensors to have dimension of at least 2. Got shape: {right.ContentString()}");
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
                if (dimSize == -1) {
                    return null;
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
