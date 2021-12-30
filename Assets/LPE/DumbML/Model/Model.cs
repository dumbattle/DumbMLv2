using System.Collections.Generic;
using System.Linq;
using System;


namespace DumbML {
    public class Model {
        ModelNode[] inputNodes;
        ModelNode[] outputNodes;
        //ModelNode[] lossNodes;
        List<ModelNode> allNodes;
        IReadOnlyList<ITensorBuffer> outputBuffers;

        Gradients gradients;
        Optimizer optimizer;

        Model backwardsModel;
        public Model(InputOp[] inputs, Operation[] outputs) {
            Dictionary<Operation, ModelNode> op2node = new Dictionary<Operation, ModelNode>();
            Stack<Operation> opStack = new Stack<Operation>();
            allNodes = new List<ModelNode>();
            foreach (var outputOp in outputs) {
                opStack.Push(outputOp);

                while (opStack.Count > 0) {
                    var op = opStack.Pop();
                    var n = BuildNode(op);
                }
            }


            outputNodes = new ModelNode[outputs.Length];
            inputNodes = new ModelNode[inputs.Length];


            for (int i = 0; i < outputs.Length; i++) {
                outputNodes[i] = op2node[outputs[i]];
            }

            for (int i = 0; i < inputs.Length; i++) {
                inputNodes[i] = op2node[inputs[i]];
            }

            outputBuffers = Array.AsReadOnly((from x in outputNodes select x.outputBuffer).ToArray());

            ModelNode BuildNode(Operation op) {
                if (op2node.ContainsKey(op)) {
                    return op2node[op];
                }

                var innerOps = op.inner;
                var innerNodes = new ModelNode[innerOps.Length];

                for (int i = 0; i < op.inner.Length; i++) {
                    Operation innerOp = op.inner[i];

                    innerNodes[i] = BuildNode(innerOp);
                }

                var n = new ModelNode(op, innerNodes);
                op2node.Add(op, n);
                allNodes.Add(n);

                return op2node[op];
            }
        }

        public IReadOnlyList<ITensorBuffer> Call(params FloatTensor[] inputs) {
            if (inputs.Length != inputNodes.Length) {
                throw new ArgumentException($"Worng number of inputs received\n  Expected: {inputNodes.Length}\n  Got: {inputs.Length}");
            }

            // set input nodes
            for (int i = 0; i < inputs.Length; i++) {
                inputNodes[i].outputBuffer.SetShape(inputs[i].shape);
                inputNodes[i].outputBuffer.CopyFrom(inputs[i]);
            }

            // call each node
            foreach (var n in allNodes) {
                n.Forward();
            }

            return outputBuffers;
        }
      
        public void Backwards() {
            var grads = backwardsModel.Call();
            optimizer.ZeroGrad();
            for (int i = 0; i < grads.Count; i++) {
                var g = grads[i];
                var k = gradients.keys[i];
                BLAS.Engine.Compute.Add(gradients[k], g, gradients[k]);

            }
            optimizer.Update();
            //foreach (var n in allNodes) {
            //    n.ClearError();
            //}

            //foreach (var n in lossNodes) {
            //    BLAS.Engine.Compute.SetTo1s(n.errorBuffer);
            //}

            //optimizer.ZeroGrad();
            //for (int i = allNodes.Count - 1; i >= 0; i--) {
            //    allNodes[i].Backwards(gradients);
            //}
            //optimizer.Update();
        }

        public void CreateBackwardsModel(Operation[] grads, Operation[] wrt) {

            Dictionary<Operation, Operation> src = allNodes.ToDictionary(x => x.op, x => (Operation)new BufferOp(x.outputBuffer));
            Dictionary<Operation, Operation> errors = allNodes.ToDictionary(x => x.op, x => (Operation)null);

            //foreach (var op in wrt) {
            //    Operation seed = new Ones(op.shape);
            //    errors[op] = seed;
            //}

            for (int i = allNodes.Count - 1; i >= 0; i--) {
                Operation op = allNodes[i].op;
                Operation[] inputs = (from x in op.inner select src[x]).ToArray();

                Operation err = errors[op];
                if (err == null) {
                    Operation seed = new Ones(op.shape);
                    errors[op] = seed;
                    err = seed;
                }
                var inputGrads = op.BuildBackwards(inputs, src[op], err);

                for (int j = 0; j < inputs.Length; j++) {
                    var e = errors[op.inner[j]];
                    var g = inputGrads[j];
                    if(g == null) {
                        continue;
                    }
                    e = e == null ? g : new Add(e, g);
                    errors[op.inner[j]] = e;
                }
            }
            Operation[] outputOps = (from x in grads select errors[x]).ToArray();
            backwardsModel = new Model(new InputOp[0], outputOps);
        }

        public void InitTraining(Optimizer o, params Operation[] loss) {
            if (!o.IsBuilt) {
                List<Variable> vars = new List<Variable>();
                foreach (var n in allNodes) {
                    if (n.op is Variable v) {
                        vars.Add(v);
                    }
                }
                var g = new Gradients(vars);
                o.InitializeGradients(g);
            }


            //lossNodes = (from x in loss select (from y in allNodes where y.op == x select y).First()).ToArray();
            optimizer = o;
            gradients = o.grad;
            CreateBackwardsModel(o.grad.keys, loss);
        }

        public void Dispose() {
            foreach (var n in allNodes) {
                n.Dispose();
            }

            gradients?.Dispose();
            optimizer?.Dispose();
            backwardsModel?.Dispose();
            inputNodes = null;
            outputNodes = null;
            allNodes = null;
            gradients = null;
            optimizer = null;
        }

        class BufferOp : Operation {
            ITensorBuffer buffer; // not owner of this buffer, so no need to dispose

            public BufferOp(ITensorBuffer buffer) {
                this.buffer = buffer;
                BuildOp(buffer.shape, buffer.dtype);
            }

            public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
                result.SetShape(buffer.shape);
                BLAS.Engine.Compute.Copy(buffer, result);
            }
        
            public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
                return new Operation[] {
                    new Multiply(error, inputs[1]),
                    new Multiply(error, inputs[0])
                };
            }

        }
    }
}
