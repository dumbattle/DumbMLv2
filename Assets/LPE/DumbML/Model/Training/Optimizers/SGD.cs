using System.Collections.Generic;


namespace DumbML {
    public class SGD : Optimizer {
        float lr;
        float momentum;
        Dictionary<Variable, ITensorBuffer> momentumBuffer = new Dictionary<Variable, ITensorBuffer>();
        Dictionary<Variable, ITensorBuffer> temp1Buffer = new Dictionary<Variable, ITensorBuffer>();


        public SGD(Gradients g, float lr = .01f, float momentum = .9f) : base(g) {
            this.lr = lr;
            this.momentum = momentum;
        }
        public SGD(float lr = .01f, float momentum = .9f) : base() {
            this.lr = lr;
            this.momentum = momentum;
        }

        public override void InitializeGradients(Gradients g) {
            base.InitializeGradients(g);
            momentumBuffer = CreateBufferDict(g);
            temp1Buffer = CreateBufferDict(g);
        }
        
        
        public override void UpdateWeight(Variable variable, ITensorBuffer grad, ITensorBuffer variableBuffer) {
            var mBuf = momentumBuffer[variable];
            var t1 = temp1Buffer[variable];

            // scale grad
            BLAS.Engine.Compute.Multiply(grad, (1 - momentum) * lr, t1);
            // scale momentum
            BLAS.Engine.Compute.Multiply(mBuf, momentum, mBuf);
            // update momentum
            BLAS.Engine.Compute.Add(t1, mBuf, mBuf);
            // update weight
            BLAS.Engine.Compute.Subtract(variableBuffer, mBuf, t1);
            BLAS.Engine.Compute.Copy(t1, variableBuffer);
        }


        public override void Dispose() {
            base.Dispose();
            DisposeBufferDict(momentumBuffer);
            DisposeBufferDict(temp1Buffer);
        } 
    }
}
