namespace DumbML {
    public class InputOp : Operation {
        public InputOp(DType dtype, params int[] shape) {
            BuildOp(shape, dtype);
        }

        public InputOp(params int[] shape) {
            BuildOp(shape, DType.Float);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) { }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return null;
        }
    }
}
