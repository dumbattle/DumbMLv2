namespace DumbML {
    public class Square : Operation {
        public Square(Operation a) {
            BuildOp(a.shape, DType.Float, a);
        }


        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            result.SetShape(inputs[0].shape);
            BLAS.Engine.Compute.Square(inputs[0], result);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            Operation result = new Multiply(error, inputs[0]);
            return new Operation[] {
                new Multiply(result, 2)
            };
        }
    }
}
