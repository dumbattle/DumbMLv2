namespace DumbML {
    public class OneHot : Operation {
        int[] _shape;
        int depth; 
        float on;
        float off;


        public OneHot(Operation op, int depth, float on = 1, float off = 0) { 
            this.depth = depth;
            this.on = on;
            this.off = off;


            _shape = GetShape(op.shape, depth, _shape);

            BuildOp(_shape, DType.Float, op);
        }
        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            _shape = GetShape(inputs[0].shape, depth, _shape);
            BLAS.Engine.Compute.OneHot(inputs[0], depth, on, off, result);
        }
        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return null;
        }

        static int[] GetShape(int[] inputShape, int depth, int[] result) {
            result = result ?? new int[inputShape.Length + 1];
            for (int i = 0; i < inputShape.Length; i++) {
                result[i] = inputShape[i];
            }
            result[result.Length - 1] = depth;
            return result;
        }
    }
}
