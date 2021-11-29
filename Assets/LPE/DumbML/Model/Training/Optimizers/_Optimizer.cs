using System.Collections.Generic;


namespace DumbML {
    public abstract class Optimizer {
        public Gradients grad { get; protected set; }
        public bool IsBuilt { get; protected set; }

        Dictionary<Variable, ITensorBuffer> weightBuffers = new Dictionary<Variable, ITensorBuffer>();



        public Optimizer(Gradients grad) {
            InitializeGradients(grad);
        }
        public Optimizer() {
            IsBuilt = false;

        }


        public virtual void InitializeGradients(Gradients g) {
            grad = g;
            IsBuilt = true;

            weightBuffers = CreateBufferDict(g);
            foreach (var (v, b) in weightBuffers) {
                b.CopyFrom(v.value);
            }
        }


        public void ZeroGrad() {
            grad.Reset();
        }

        public virtual void Update() {
            foreach (var (v, buf) in weightBuffers) {
                // not trainable 
                if (!v.trainable) {
                    continue;
                }

                ITensorBuffer gradBuffer = grad[v];
                UpdateWeight(v, gradBuffer, buf);
                buf.CopyTo(v.value);
            }
        }

        public abstract void UpdateWeight(Variable variable, ITensorBuffer grad, ITensorBuffer variableBuffer);

        public virtual void Dispose() {
            DisposeBufferDict(weightBuffers);
        }
    
        protected void DisposeBufferDict(Dictionary<Variable, ITensorBuffer> d) {
            foreach (var (_, buf) in d) {
                buf.Dispose();
            }
        }
        protected Dictionary<Variable, ITensorBuffer> CreateBufferDict(Gradients g) {
            Dictionary<Variable, ITensorBuffer> result = new Dictionary<Variable, ITensorBuffer>();

            foreach (var op in g.keys) {
                if (op is Variable v) {
                    var buf = BLAS.Engine.GetTensorBuffer(v.shape);
                    BLAS.Engine.Compute.Clear(buf);
                    result.Add(v, buf);
                }
            }

            return result;
        }
    }
}
