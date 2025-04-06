namespace DumbML {
    public class ReLU : Operation {
        public ReLU(Operation op) {
            BuildOp(op.shape, op.dtype, op);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            result.SetShape(inputs[0].shape);
            BLAS.Engine.Compute.ReLU(inputs[0], result);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            // which elements where not relu'd
            Operation op = new ElementwiseEquals(inputs[0], output);

            // 1 if not relu'd, 0 if relu'd
            op = new Cast(op, DType.Float);

            // no gradient for relu'd elements
            op = new Multiply(error, op);

            return new Operation[] {
                op
            };
        }
    }

}
