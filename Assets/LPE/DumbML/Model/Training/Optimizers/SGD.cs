using System.Collections.Generic;
using static DumbML.BLAS.CPU.ElementwiseFloatParamJobs;


namespace DumbML {
    public class RMSProp : Optimizer {
        Dictionary<Variable, ITensorBuffer> vBuffer = new Dictionary<Variable, ITensorBuffer>();
        Dictionary<Variable, ITensorBuffer> temp1Buffer = new Dictionary<Variable, ITensorBuffer>();
        float lr;
        float gamma;


        public RMSProp(Gradients g, float lr = .001f, float gamma = .999f) : base(g) {
            this.lr = lr;
            this.gamma = gamma;
        }
        public RMSProp(float lr = .001f, float gamma = .999f) : base() {
            this.lr = lr;
            this.gamma = gamma;
        }

        public override void InitializeGradients(Gradients g) {
            base.InitializeGradients(g);
            vBuffer = CreateBufferDict(g);
            temp1Buffer = CreateBufferDict(g);
        }


        public override void UpdateWeight(Variable variable, ITensorBuffer grad, ITensorBuffer variableBuffer) {
            var v = vBuffer[variable];
            var t1 = temp1Buffer[variable];

            //v = v * gamma + g * g * (1 - gamma);
            BLAS.Engine.Compute.Multiply(v, gamma, v);       // v * gamma
            BLAS.Engine.Compute.Multiply(grad, grad, t1);    // g * g
            BLAS.Engine.Compute.Multiply(t1, 1 - gamma, t1); // g * g * (1 - gamma)
            BLAS.Engine.Compute.Add(v, t1, v);               // v = v * gamma + g * g * (1 - gamma);

            // result = g / sqrt(v + 1e-5f);
            BLAS.Engine.Compute.Add(v, t1, 1e-5f);    // v + 1e-5f
            BLAS.Engine.Compute.SquareRoot(t1, t1);   // sqrt(v + 1e-5f)
            BLAS.Engine.Compute.Divide(grad, t1, t1); // g /  sqrt(v + 1e-5f)

            //if (result > .0001f) {
            //    result = .0001f;
            //}
            //else if (result < -.0001f) {
            //    result = -.0001f;
            //}
            BLAS.Engine.Compute.Min(t1,  0.0001f, t1);
            BLAS.Engine.Compute.Max(t1, -0.0001f, t1);


            //result = result * lr;
            BLAS.Engine.Compute.Multiply(t1, lr, t1);


            // update weight
            // variableBuffer = variableBuffer - t1
            BLAS.Engine.Compute.Subtract(variableBuffer, t1, variableBuffer);
        }
        public override void Dispose() {
            base.Dispose();
            DisposeBufferDict(vBuffer);
            DisposeBufferDict(temp1Buffer);
        }
    }


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
            // t1 = grad * lr
            BLAS.Engine.Compute.Multiply(grad, lr, t1);
            // scale momentum
            // mBuf = mBuf * momentum
            BLAS.Engine.Compute.Multiply(mBuf, momentum, mBuf);
            // update momentum
            // mBuf = t1 + mBuf
            BLAS.Engine.Compute.Add(t1, mBuf, mBuf);
            // update weight
            // variableBuffer = variableBuffer - mBuf
            BLAS.Engine.Compute.Subtract(variableBuffer, mBuf, variableBuffer);
        }


        public override void Dispose() {
            base.Dispose();
            DisposeBufferDict(momentumBuffer);
            DisposeBufferDict(temp1Buffer);
        } 
    }
}
