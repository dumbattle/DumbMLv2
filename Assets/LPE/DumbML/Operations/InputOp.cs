namespace DumbML {
    public class InputOp : Operation {
        public InputOp(DType dtype, params int[] shape) {
            BuildOp(shape, dtype);
        }

        public InputOp(params int[] shape) {
            BuildOp(shape, DType.Float);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            if (inputs.Length == 0) {
                return;
            }
            result.SetShape(inputs[0].shape);
            BLAS.Engine.Compute.Copy(inputs[0], result);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return null;
        }

        public void SetSource(Operation src) {
            BuildOp(shape, dtype, src);
        }
    }
}
