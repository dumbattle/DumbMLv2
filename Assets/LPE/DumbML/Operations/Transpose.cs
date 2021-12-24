namespace DumbML {
    public class Transpose : Operation {
        int[] perm;
        int[] shapeActual;

        public Transpose(Operation op, params int[] permutation ) {
            perm = (int[])perm.Clone();
            var s = GetShape(op.shape, perm, null);
            BuildOp(s, op.dtype, op);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            shapeActual = GetShape(inputs[0].shape, perm, shapeActual);
            result.SetShape(shapeActual);

            BLAS.Engine.Compute.Transpose(inputs[0], perm, result);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return new Operation[] { 
                new Transpose(error, GetReverseTranspose(perm))
            };
        }

        static int[] GetShape(int[] src, int[] perm, int[] result) {
            result = result ?? new int[src.Length];
            for (int i = 0; i < perm.Length; i++) {
                result[i] = src[perm[i]];
            }
            return result;
        }

        static int[] GetReverseTranspose(int[] perm) {
            var result = new int[perm.Length];
            for (int i = 0; i < result.Length; i++) {
                result[perm[i]] = i;
            }
            return result;
        }
    }
}
