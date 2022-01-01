using System.Collections.Generic;


namespace DumbML {
    public class ReduceSum : Operation {
        int[] axis;

        public ReduceSum(Operation op, params int[] axis) {
            if (axis == null) {
                BuildOp(new int[] { 1 }, op.dtype, op);
            }
            else {
                this.axis = (int[])axis.Clone();

                // determine output shape
                List<int> shapeList = new List<int>();

                // add non-reduced dimensions
                for (int i = 0; i < op.shape.Length; i++) {
                    bool isReduced = false;
                    foreach (var a in axis) {
                        if (a == i) {
                            isReduced = true;
                            break;
                        }
                    }
                    if (!isReduced) {
                        shapeList.Add(op.shape[i]);
                    }
                }

                // if all dimensions are reduced, set shape to [1]
                if (shapeList.Count == 0) {
                    shapeList.Add(1);
                }

                BuildOp(shapeList.ToArray(), op.dtype, op);
            }
        }

        /// <summary>
        /// output shape may not be exactly correct, reshaping may be necessary
        /// </summary>
        public ReduceSum(Operation op, Operation matchShape) {
            BuildOp(matchShape.shape, op.dtype, op, matchShape);
            axis = new int[op.shape.Length];
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            if (inputs.Length == 2) {
                GetReductionAxis(inputs[0].shape, inputs[1].shape, axis);
            }

            BLAS.Engine.Compute.ReduceSum(inputs[0], axis, result);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            int[] a = axis;

            if (a == null) {
                a = new int[inputs[0].shape.Length - 1];
                for (int i = 0; i < a.Length; i++) {
                    a[i] = i;
                }
            }

            Operation op = new AddDims(error, a);
            op = new Broadcast(op, inputs[0]);

            return new Operation[] {
               op
           };
        }

        static void GetReductionAxis(int[] shape, int[] target, int[] result) {
            // clear results
            for (int i = 0; i < result.Length; i++) {
                result[i] = shape.Length;
            }

            for (int i = shape.Length; i > 0; i--) {
                int si = shape.Length - i;
                int ti = target.Length - i;

                if (ti < 0) {
                    result[si] = si;
                    continue;
                }

                int sdim = shape[si];
                int tdim = target[ti];

                if (tdim == 1 && sdim > 1) {
                    result[si] = si;
                    continue;
                }
            }
        }
    }

}
