namespace DumbML {
    public class Log : Operation {
        public Log(Operation op) {
            BuildOp(op.shape, op.dtype, op);
        }
        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            result.SetShape(inputs[0].shape);
            BLAS.Engine.Compute.Log(inputs[0], result);
        }
        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return new[] {
                new Divide(error, inputs[0])
            };
        }
    }
}
