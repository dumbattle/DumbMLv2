namespace DumbML {
    public class NoGrad : Operation {
        public NoGrad(Operation op) {
            BuildOp(op.shape, op.dtype, op);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            result.SetShape(inputs[0].shape);
            BLAS.Engine.Compute.Copy(inputs[0], result, true);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return null;
        }
    }
}
