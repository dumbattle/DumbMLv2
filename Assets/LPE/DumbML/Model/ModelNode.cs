namespace DumbML {
    public class ModelNode {
        public Operation op { get; private set; }

        public ITensorBuffer outputBuffer;

        ITensorBuffer[] inputBuffers;

        public ModelNode(Operation op, ModelNode[] inputNodes) {
            this.op = op;

            outputBuffer = BLAS.Engine.GetTensorBuffer(op.dtype, op.shape);

            inputBuffers = new ITensorBuffer[inputNodes.Length];

            for (int i = 0; i < inputNodes.Length; i++) {
                inputBuffers[i] = inputNodes[i].outputBuffer;
            }
        }

        public void Forward() {
            op.Forward(inputBuffers, outputBuffer);
        }

        public void Dispose() {
            outputBuffer.Dispose();
        }
    }
}
