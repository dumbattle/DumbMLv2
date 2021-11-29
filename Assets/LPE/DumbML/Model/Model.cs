using System.Collections.Generic;
using System.Linq;
using System;


namespace DumbML {
    public class Model {
        ModelNode[] inputNodes;
        ModelNode[] outputNodes;
        ModelNode[] lossNodes;
        List<ModelNode> allNodes;
        IReadOnlyList<ITensorBuffer> outputBuffers;

        Gradients gradients;
        Optimizer optimizer;

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




        public IReadOnlyList<ITensorBuffer> Call(params Tensor[] inputs) {
            if (inputs.Length != inputNodes.Length) {
                throw new ArgumentException($"Worng number of inputs received\n  Expected: {inputNodes.Length}\n  Got: {inputs.Length}");
            }

            // set input nodes
            for (int i = 0; i < inputs.Length; i++) {
                inputNodes[i].outputBuffer.CopyFrom(inputs[i]);
            }

            // call each node
            foreach (var n in allNodes) {
                n.Forward();
            }

            return outputBuffers;
        }

        public void Backwards() {
            foreach (var n in allNodes) {
                n.ClearError();
            }

            foreach (var n in lossNodes) {
                BLAS.Engine.Compute.SetTo1s(n.errorBuffer);
            }

            optimizer.ZeroGrad();
            for (int i = allNodes.Count - 1; i >= 0; i--) {
                allNodes[i].Backwards(gradients);
            }
            optimizer.Update();
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


            lossNodes = (from x in loss select (from y in allNodes where y.op == x select y).First()).ToArray();
            optimizer = o;
            gradients = o.grad;
        }

        public void Dispose() {
            foreach (var n in allNodes) {
                n.Dispose();
            }

            gradients?.Dispose();
            optimizer?.Dispose();

            inputNodes = null;
            outputNodes = null;
            allNodes = null;
            gradients = null;
            optimizer = null;
        }
    }
}
