namespace DumbML {
    public class ReduceSum : Operation {
        int[] axis;

        public ReduceSum(Operation op, params int[] axis) {
            if (axis == null) {
                BuildOp(new int[] { 1 }, op.dtype, op);

            }
            else {

                this.axis = (int[])axis.Clone();
                int size = op.shape.Length - axis.Length;
                if (size == 0) {
                    size = 1;
                }
                int[] shape = new int[size];


                int ind = 0;
                for (int i = 0; i < op.shape.Length; i++) {
                    bool isReduced = false;
                    foreach (var a in axis) {
                        if (a == i) {
                            isReduced = true;
                            break;
                        }
                    }
                    if (!isReduced) {
                        shape[ind] = op.shape[i];
                        ind++;
                    }
                }

                if (size == 0) {
                    shape[0] = 1;
                }

                BuildOp(shape, op.dtype, op);
            }
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            BLAS.Engine.Compute.ReduceSum(inputs[0], axis, result);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return new Operation[] {
               null
           };
        }
    }
}
