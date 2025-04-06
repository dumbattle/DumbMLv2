namespace DumbML {
    public class ReduceSum : Operation {
        int[] axis;
        int[] _shapeActual;

        public ReduceSum(Operation op, params int[] axis) {
            // TODO validate axis
            this.axis = (int[])axis?.Clone() ?? new int[0];

            HandleNegatives(this.axis, op.shape.Length);
            BuildOp(GetOutputShape(op.shape, this.axis, null), op.dtype, op);
        }

        /// <summary>
        /// output shape may not be exactly correct, reshaping may be necessary.  
        /// </summary>
        public ReduceSum(Operation op, Operation matchShape) {
            BuildOp(matchShape.shape, op.dtype, op, matchShape);
            axis = GetReductionAxis(op.shape, matchShape.shape);
        }

        //**********************************************************************************************************************
        // Op Implementation
        //**********************************************************************************************************************

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            if (inputs.Length == 2) {
                result.SetShape(inputs[1].shape);
            }
            else {
                _shapeActual = GetOutputShape(inputs[0].shape, axis, _shapeActual);
                result.SetShape(_shapeActual);
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

        //**********************************************************************************************************************
        // Helper Methods
        //**********************************************************************************************************************

        static void HandleNegatives(int[] axis, int rank) {
            for (int i = 0; i < axis.Length; i++) {
                if (axis[i] < 0) {
                    axis[i] = rank + axis[i];
                }
            }
        }
        static int[] GetOutputShape(int[] inputShape, int[] axis, int[] result) {
            int dimCount = inputShape.Length - axis.Length;

            if (axis == null || dimCount == 0 || axis.Length == 0) {
                result = result ?? new int[] { 1 };
                result[0] = 1;
            }
            else {
                // determine output shape
                result = result ?? new int[dimCount];

                // add non-reduced dimensions
                int dim = 0;
                for (int i = 0; i < inputShape.Length; i++) {
                    bool isReduced = false;
                    foreach (var a in axis) {
                        if (a == i) {
                            isReduced = true;
                            break;
                        }
                    }
                    if (!isReduced) {
                        result[dim] = (inputShape[i]);

                        dim++;
                    }
                }
            }

            return result;
        }

        static int[] GetReductionAxis(int[] shape, int[] target) {
            int[] result = new int[shape.Length];
            // clear results
            for (int i = 0; i < result.Length; i++) {
                result[i] = shape.Length;
            }
            int validIndices = 0;

            for (int i = shape.Length; i > 0; i--) {
                int si = shape.Length - i;
                int ti = target.Length - i;

                if (ti < 0) {
                    result[validIndices] = si;
                    validIndices++;
                    continue;
                }

                int sdim = shape[si];
                int tdim = target[ti];

                if (tdim == 1 && sdim > 1) {
                    result[validIndices] = si;
                    validIndices++;
                    continue;
                }
            }

            var r2 = new int[validIndices];

            for (int i = 0; i < r2.Length; i++) {
                r2[i] = result[i];
            }

            return r2;
        }
    }
}
