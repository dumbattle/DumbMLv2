namespace DumbML {
    public class Ones : Operation {
        public Ones(params int[] shape) {
            BuildOp(shape, DType.Float);
        }
        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            BLAS.Engine.Compute.SetTo1s(result);
        }
        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return null;
        }
    }


}
