namespace DumbML {
    public class ElementwiseEquals : Operation {
        int[] shapeActual;


        public ElementwiseEquals(Operation a, Operation b) {
            shapeActual = OpUtility.GetBroadcastShape(a.shape, b.shape, shapeActual);
            BuildOp(shapeActual, DType.Bool, a, b);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            shapeActual = OpUtility.GetBroadcastShape(inputs[0].shape, inputs[1].shape, shapeActual);
            result.SetShape(shapeActual);
            BLAS.Engine.Compute.ElementwiseEquals(inputs[0], inputs[1], result);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            // no gradients
            return null;
        }
    }

}
