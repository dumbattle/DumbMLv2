namespace DumbML {
    public class Multiply : Operation {
        public Multiply(Operation a, Operation b) {
            if (!ShapeUtility.SameShape(a.shape, b.shape)) {
                throw new System.ArgumentException($"Cannot multiply 2 Tensors of different shapes. {a.shape.ContentString()} - {b.shape.ContentString()}");
            }

            BuildOp(a.shape, DType.Float, a, b);
        }


        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            BLAS.Engine.Compute.Multiply(inputs[0], inputs[1], result);
        }


        public override void Backward(ITensorBuffer[] inputs, ITensorBuffer output, ITensorBuffer error, ITensorBuffer[] results) {
            BLAS.Engine.Compute.Multiply(error, inputs[1], results[0]);
            BLAS.Engine.Compute.Multiply(error, inputs[0], results[1]);
        }
    }
}
