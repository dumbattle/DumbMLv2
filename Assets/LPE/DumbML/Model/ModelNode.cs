namespace DumbML {
    public class ModelNode {
        public Operation op { get; private set; }

        public ITensorBuffer outputBuffer;
        //public ITensorBuffer errorBuffer;


        ITensorBuffer[] inputBuffers;
        //ITensorBuffer[] inputErrorBuffers; // ref to actual error in the node
        //ITensorBuffer[] inputErrorTempBuffers; // multiple ops may contribute to error

        public ModelNode(Operation op, ModelNode[] inputNodes) {
            this.op = op;

            outputBuffer = BLAS.Engine.GetTensorBuffer(op.dtype, op.shape);
            //errorBuffer = BLAS.Engine.GetTensorBuffer(op.dtype, op.shape);

            inputBuffers = new ITensorBuffer[inputNodes.Length];
            //inputErrorBuffers = new ITensorBuffer[inputNodes.Length];
            //inputErrorTempBuffers = new ITensorBuffer[inputNodes.Length];

            for (int i = 0; i < inputNodes.Length; i++) {
                inputBuffers[i] = inputNodes[i].outputBuffer;
                //inputErrorBuffers[i] = inputNodes[i].errorBuffer;
                //inputErrorTempBuffers[i] = BLAS.Engine.GetTensorBuffer(DType.Float, inputNodes[i].outputBuffer.shape);
            }
        }

        public void Forward() {
            op.Forward(inputBuffers, outputBuffer);
        }

        //public void Backwards(Gradients g) {
        //    if (g.Contains(op)) {
        //        BLAS.Engine.Compute.Add(g[op],errorBuffer, g[op]);
        //    }
        //    op.Backward(inputBuffers, outputBuffer, errorBuffer, inputErrorTempBuffers);

        //    // update cumalative errors
        //    for (int i = 0; i < inputErrorBuffers.Length; i++) {
        //        BLAS.Engine.Compute.Add(inputErrorBuffers[i], inputErrorTempBuffers[i], inputErrorBuffers[i]);
        //    }
        //}

        //public void ClearError() {
        //    BLAS.Engine.Compute.Clear(errorBuffer);
        //}

        public void Dispose() {
            //foreach (var b in inputErrorTempBuffers) {
            //    b.Dispose();
            //}
            outputBuffer.Dispose();
            //errorBuffer.Dispose();
        }
    }
}
