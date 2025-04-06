namespace DumbML {
    public class DebugPrint : Operation {
        string label;

        public DebugPrint(Operation op, string label) {
            this.label = label;
            BuildOp(op.shape, op.dtype, op);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            result.SetShape(inputs[0].shape);
            BLAS.Engine.Compute.Copy(inputs[0], result);
            Tensor t = Tensor.Get(dtype, inputs[0].shape);
            inputs[0].CopyTo(t);
            UnityEngine.Debug.Log(label + ": " + t);
            if (t is FloatTensor ft) {
                if (float.IsNaN(ft.data[0])) {
                    UnityEngine.Debug.Break();
                }
            }
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            // error is always float
            // no need to recast
            return new Operation[] {
                new DebugPrint(error, label + "-Backwards") 
            };
        }
    }
}
