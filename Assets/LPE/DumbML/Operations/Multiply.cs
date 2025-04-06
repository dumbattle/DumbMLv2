using System.Collections.Generic;

namespace DumbML {
    public class Multiply : Operation {
        int[] shapeActual;

        public Multiply(Operation a, Operation b) {
            shapeActual = OpUtility.GetBroadcastShape(a.shape, b.shape, shapeActual);
            BuildOp(shapeActual, DType.Float, a, b);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            shapeActual = OpUtility.GetBroadcastShape(inputs[0].shape, inputs[1].shape, shapeActual);
            result.SetShape(shapeActual);

            BLAS.Engine.Compute.Multiply(inputs[0], inputs[1], result);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            List<int> a = OpUtility.BroadcastBackwardsReductionShape(inputs[0].shape, error.shape);
            List<int> b = OpUtility.BroadcastBackwardsReductionShape(inputs[1].shape, error.shape);

            var agrad = a.Count > 0 ? new Reshape(new ReduceSum(error, inputs[0]), inputs[0]) : error;
            var bgrad = b.Count > 0 ? new Reshape(new ReduceSum(error, inputs[1]), inputs[1]) : error;
            return new Operation[] {
                new Multiply(agrad, inputs[1]),
                new Multiply(bgrad, inputs[0])
            };
        }
    }
}
