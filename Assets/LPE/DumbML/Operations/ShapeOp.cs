namespace DumbML {
    public class ShapeOp : Operation {
        IntTensor value;

        public ShapeOp(Operation op) {
            value = new IntTensor(op.shape.Length);
            for (int i = 0; i < value.size; i++) {
                value[i] = op.shape[i];
            }

            BuildOp(value.data, DType.Int, op);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            for (int i = 0; i < value.size; i++) {
                value[i] = inputs[0].shape[i];
            }

            result.CopyFrom(value);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return new Operation[] { null };
        }
    }

    //public class Reshape : Operation {
    //    public Reshape(Operation op, params int[] shape) {
    //        BuildOp(shape, op.dtype, op);
    //    }


    //    public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
    //    }
    //    public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
    //    }
    //}
}
