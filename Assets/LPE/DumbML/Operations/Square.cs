namespace DumbML {
    public class Square : Operation {
        public Square(Operation a) {
            BuildOp(a.shape, DType.Float, a);
        }


        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            BLAS.Engine.Compute.Square(inputs[0], result);
        }


        public override void Backward(ITensorBuffer[] inputs, ITensorBuffer output, ITensorBuffer error, ITensorBuffer[] results) {
            BLAS.Engine.Compute.Multiply(inputs[0], error, results[0]);
            BLAS.Engine.Compute.Multiply(results[0], 2, results[0]);
        }
    }
}
