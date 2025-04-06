using System.Collections.Generic;
using System.Linq;


namespace DumbML {
    public class Add : Operation {
        int[] shapeActual;

        public Add(Operation a, Operation b) {
            shapeActual = OpUtility.GetBroadcastShape(a.shape, b.shape, shapeActual);
            BuildOp(shapeActual, DType.Float, a, b);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            shapeActual = OpUtility.GetBroadcastShape(inputs[0].shape, inputs[1].shape, shapeActual);
            result.SetShape(shapeActual);

            BLAS.Engine.Compute.Add(inputs[0], inputs[1], result);
        }
   
        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            List<int> a = OpUtility.BroadcastBackwardsReductionShape(inputs[0].shape, error.shape);
            List<int> b = OpUtility.BroadcastBackwardsReductionShape(inputs[1].shape, error.shape);


            return new Operation[] {
                a.Count > 0 ? new Reshape(new ReduceSum(error, inputs[0]), inputs[0]) : error,
                b.Count > 0 ? new Reshape(new ReduceSum(error, inputs[1]), inputs[1]) : error
            };
        }
    }
}
