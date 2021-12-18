namespace DumbML {
    public class Subtract : Operation {
        public Subtract(Operation a, Operation b) {
            if (!ShapeUtility.SameShape(a.shape, b.shape)) {
                throw new System.ArgumentException($"Cannot subtract 2 Tensors of different shapes. {a.shape.ContentString()} - {b.shape.ContentString()}");
            }

            BuildOp(a.shape, DType.Float, a, b);
        }


        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            BLAS.Engine.Compute.Subtract(inputs[0], inputs[1], result);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return new Operation[] {
                error,
                new Multiply(error, -1)
            };
        }
    }
}
