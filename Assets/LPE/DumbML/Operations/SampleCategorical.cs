namespace DumbML {
    public class SampleCategorical : Operation {
        int[] _shape;

        public SampleCategorical (Operation op) {
            _shape = new int[op.shape.Length];
            for (int i = 0; i < _shape.Length; i++) {
                _shape[i] = op.shape[i];
            }
            _shape[_shape.Length - 1] = 1;
            BuildOp(_shape, DType.Int, op);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            for (int i = 0; i < _shape.Length - 1; i++) {
                _shape[i] = inputs[0].shape[i];
            }
            _shape[_shape.Length - 1] = 1;
            result.SetShape(_shape);
            BLAS.Engine.Compute.SampleCategorical(inputs[0], result);
        }
        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return null;
        }
    }
}
