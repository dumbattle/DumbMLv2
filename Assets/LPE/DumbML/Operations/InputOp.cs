namespace DumbML {
    public class InputOp : Operation {
        public InputOp(params int[] shape) {
            BuildOp(shape);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) { }

        public override void Backward(ITensorBuffer[] inputs, ITensorBuffer output, ITensorBuffer error, ITensorBuffer[] results) { }

    }
}
