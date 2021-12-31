namespace DumbML {
    public class Zeros : Operation {
        public Zeros(params int[] shape) {
            BuildOp(shape, DType.Float);
        }
        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            BLAS.Engine.Compute.SetTo0s(result);
        }
        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return null;
        }
    }


}
