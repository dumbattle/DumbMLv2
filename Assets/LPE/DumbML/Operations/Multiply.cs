namespace DumbML {
    public class Multiply : Operation {
        public Multiply(Operation a, Operation b) {
            BuildOp(a.shape, DType.Float, a, b);
        }


        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            BLAS.Engine.Compute.Multiply(inputs[0], inputs[1], result);
        }


        public override void Backward(ITensorBuffer[] inputs, ITensorBuffer output, ITensorBuffer error, ITensorBuffer[] results) {
            BLAS.Engine.Compute.Multiply(error, inputs[1], results[0]);
            BLAS.Engine.Compute.Multiply(error, inputs[0], results[1]);
        }
        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return new Operation[] {
                new Multiply(error, inputs[1]),
                new Multiply(error, inputs[0])
            };
        }
    }
}
