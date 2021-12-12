namespace DumbML {
    public class Add : Operation {
        public Add(Operation a, Operation b) {
            if (!ShapeUtility.SameShape(a.shape, b.shape)) {
                throw new System.ArgumentException($"Cannot add 2 Tensors of different shapes. {a.shape.ContentString()} - {b.shape.ContentString()}");
            }

            BuildOp(a.shape, DType.Float, a, b);
        }


        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            BLAS.Engine.Compute.Add(inputs[0], inputs[1], result);
        }


        public override void Backward(ITensorBuffer[] inputs, ITensorBuffer output, ITensorBuffer error, ITensorBuffer[] results) {
            BLAS.Engine.Compute.Copy(error, results[0]);
            BLAS.Engine.Compute.Copy(error, results[1]);
        }
        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return new Operation[] {
                error,
                error
            };
        }
    }
}
