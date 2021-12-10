namespace DumbML {
    public class InputOp : Operation {
        public InputOp(DType dtype, params int[] shape) {
            BuildOp(shape, dtype);
        }

        public InputOp(params int[] shape) {
            BuildOp(shape, DType.Float);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) { }

        public override void Backward(ITensorBuffer[] inputs, ITensorBuffer output, ITensorBuffer error, ITensorBuffer[] results) { }

    }
}
