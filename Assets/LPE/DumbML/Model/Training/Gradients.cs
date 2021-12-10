using System.Collections.Generic;
using System.Linq;


namespace DumbML {
    public class Gradients {
        Dictionary<Operation, ITensorBuffer> grad = new Dictionary<Operation, ITensorBuffer>();
        public readonly Operation[] keys;

        public ITensorBuffer this[Operation key] {
            get {
                return grad[key];
            }
        }

        public Gradients(params Operation[] wrt) {
            foreach (var op in wrt) {
                if (!grad.ContainsKey(op)) {
                    grad.Add(op, BLAS.Engine.GetTensorBuffer(DType.Float, op.shape));
                }
            }
            keys = grad.Keys.ToArray();
        }

        public Gradients(IEnumerable<Operation> wrt) {
            foreach (var op in wrt) {
                if (!grad.ContainsKey(op)) {
                    grad.Add(op, BLAS.Engine.GetTensorBuffer(DType.Float,op.shape));
                }
            }
            keys = grad.Keys.ToArray();
        }
        
        public void Reset() {
            foreach (var k in keys) {
                BLAS.Engine.Compute.Clear(grad[k]);
            }
        }
        
        public bool Contains(Operation op) {
            return grad.ContainsKey(op);
        }
        
        public void Dispose() {
            foreach (var (_, buf) in grad) {
                buf.Dispose();
            }
            grad = null;
        }
    }
}
