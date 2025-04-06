using System.Collections.Generic;
using System.Linq;


namespace DumbML {
    public abstract class Optimizer {
        public Gradients grad { get; protected set; }
        public bool IsBuilt { get; protected set; }

        public List<Variable> variables;



        public Optimizer(Gradients grad) {
            InitializeGradients(grad);
        }
        public Optimizer() {
            IsBuilt = false;

        }


        public virtual void InitializeGradients(Gradients g) {
            grad = g;
            IsBuilt = true;
            variables = (from x in g.keys where x is Variable select (Variable)x).ToList();
        }


        public void ZeroGrad() {
            grad.Reset();
        }

        public virtual void Update() {
            foreach (var v in variables) {
                // not trainable 
                if (!v.trainable) {
                    continue;
                }

                ITensorBuffer gradBuffer = grad[v];
                UpdateWeight(v, gradBuffer, v.buffer);
                v.MarkBufferUpdated();
            }
        }

        public abstract void UpdateWeight(Variable variable, ITensorBuffer grad, ITensorBuffer variableBuffer);

        public virtual void Dispose() {
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
                    var buf = BLAS.Engine.GetTensorBuffer(DType.Float, v.shape);
                    BLAS.Engine.Compute.Clear(buf);
                    result.Add(v, buf);
                }
            }

            return result;
        }
    }
}
