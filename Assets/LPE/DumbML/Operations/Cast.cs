namespace DumbML {
    public class Cast : Operation {

        public Cast(Operation op, DType t) {
            BuildOp(op.shape, t, op);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            BLAS.Engine.Compute.Cast(inputs[0], result);
        }
        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            // error is always float
            // no need to recast
            return new Operation[] {
                error
            };
        }
    }

}
