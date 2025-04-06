using System;

namespace DumbML {
    public class Reshape : Operation {
        int[] s;

        public Reshape(Operation op, params int[] shape) {
            s = (int[])shape.Clone();
            BuildOp(shape, op.dtype, op);
        }

        public Reshape(Operation op, Operation matchShape) {
            BuildOp(matchShape.shape, op.dtype, op, matchShape);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            if (inputs.Length == 1) {
                result.SetShape(s);
            }
            else {
                result.SetShape(inputs[1].shape);
            }

            BLAS.Engine.Compute.Copy(inputs[0], result, true);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return new Operation[] {
                new Reshape(error, inputs[0])
            };
        }
    }
}
