namespace DumbML {
    public class MatrixMult : Operation {
        public MatrixMult(Operation a, Operation b) {
            int aRank = a.shape.Length;
            int bRank = b.shape.Length;

            // Compatible matrix shapes
            if (a.shape[aRank - 1] != b.shape[bRank - 2]) {
                throw new System.ArgumentException($"Cannot MatrixMult tensors of shapes: {a.shape.ContentString()} by {b.shape.ContentString()}");
            }

            // Compatible batch dims
            int[] shape = (int[])a.shape.Clone();
            shape[shape.Length - 1] = b.shape[b.shape.Length - 1];
            BuildOp(a.shape, DType.Float, a, b);
        }


        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            BLAS.Engine.Compute.MatrixMult(inputs[0], inputs[1], result);
        }


        public override void Backward(ITensorBuffer[] inputs, ITensorBuffer output, ITensorBuffer error, ITensorBuffer[] results) {
            throw new System.NotImplementedException();

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
    }
}
