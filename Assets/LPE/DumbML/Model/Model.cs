﻿using System.Collections.Generic;
using System.Linq;
using System;


namespace DumbML {
    public class Model {
        ModelNode[] inputNodes;
        ModelNode[] outputNodes;
        List<ModelNode> allNodes;
        IReadOnlyList<ITensorBuffer> outputBuffers;

        public Gradients gradients;
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
        public Model(InputOp input, Operation[] outputs) : this(new[] { input }, outputs ){ }
        public Model(InputOp[] inputs, Operation output) : this(inputs, new[] { output }){ }
        public Model(InputOp input, Operation output) : this(new[] { input }, new[] { output }){ }

        //*********************************************************************************************************
        // Control
        //*********************************************************************************************************

        public IReadOnlyList<ITensorBuffer> Call(params Tensor[] inputs) {

            if (inputs.Length != inputNodes.Length) {
                throw new ArgumentException($"Worng number of inputs received\n  Expected: {inputNodes.Length}\n  Got: {inputs.Length}");
            }

            // set input nodes
            for (int i = 0; i < inputs.Length; i++) {
                inputNodes[i].outputBuffer.SetShape(inputs[i].shape);
                inputNodes[i].outputBuffer.CopyFrom(inputs[i]);
            }

            // call each node
            for (int i = 0; i < allNodes.Count; i++) {
                ModelNode n = allNodes[i];
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
        }

        public Model CreateBackwardsModel(Operation[] grads, Operation[] wrt) {
            // outputs of all forward nodes
            Dictionary<Operation, Operation> src = allNodes.ToDictionary(x => x.op, x => (Operation)new BufferOp(x.outputBuffer, x.op));

            // errors of all forward nodes
            Dictionary<Operation, Operation> errors = allNodes.ToDictionary(x => x.op, x => (Operation)null);

            // seed loss
            foreach (var op in wrt) {
                Operation seed = new Ones(op.shape);
                errors[op] = seed;
            }

            for (int i = allNodes.Count - 1; i >= 0; i--) {
                Operation op = allNodes[i].op;
                Operation[] inputs = (from x in op.inner select src[x]).ToArray();

                Operation err = errors[op];

                if (err == null) {
                    // not needed for backwards model
                    continue;
                }

                var inputGrads = op.BuildBackwards(inputs, src[op], err);

                if (inputGrads == null) {
                    // no grads
                    continue;
                }

                for (int j = 0; j < inputs.Length; j++) {
                    // current partial error
                    var e = errors[op.inner[j]];

                    // error from this op
                    var g = inputGrads[j];

                    if(g == null) {
                        // no gradient to input
                        continue;
                    }

                    // update or init error
                    e = e == null ? g : new Add(e, g);
                    errors[op.inner[j]] = e;
                }
            }
            Operation[] outputOps = (from x in grads select errors[x]).ToArray();
            return new Model(new InputOp[0], outputOps);
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

            optimizer = o;
            gradients = o.grad;
            backwardsModel = CreateBackwardsModel(o.grad.keys, loss);
        }

        public void Dispose(bool disposeVariables = false) {
            foreach (var n in allNodes) {
                n.Dispose();
                if (disposeVariables && n.op is Variable v) {
                    v.buffer.Dispose();
                }
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


        //*********************************************************************************************************
        // Utility
        //*********************************************************************************************************


        public List<T> GetOperations<T>() where T : Operation {
            List<T> result = new List<T>();
            foreach (var node in allNodes) {
                if (node.op is T t) {
                    result.Add(t);
                }
            }
            return result;
        }

        /// <summary>
        /// Allocates Memory
        /// </summary>
        public List<int[]> GetInputShapes() {
            List<int[]> result = new List<int[]>();

            for (int i = 0; i < inputNodes.Length; i++) {
                result.Add((int[])inputNodes[i].op.shape.Clone());
            }
            return result;
        }
        /// <summary>
        /// Allocates Memory
        /// </summary>
        public List<int[]> GetOutputShapes() {
            List<int[]> result = new List<int[]>();

            for (int i = 0; i < outputNodes.Length; i++) {
                result.Add((int[])outputNodes[i].op.shape.Clone());
            }
            return result;
        }
        
        public Operation[] GetOutputOps() {
            Operation[] result = new Operation[outputNodes.Length];

            for (int i = 0; i < outputNodes.Length; i++) {
                result[i] = outputNodes[i].op;
            }
            return result;
        }
        public InputOp[] GetInputOps() {
            InputOp[] result = new InputOp[outputNodes.Length];

            for (int i = 0; i < inputNodes.Length; i++) {
                result[i] = (InputOp)inputNodes[i].op;
            }
            return result;
        }
        class BufferOp : Operation {
            ITensorBuffer buffer; // not owner of this buffer, so no need to dispose
            Operation op;
            public BufferOp(ITensorBuffer buffer, Operation op) {
                this.buffer = buffer;
                this.op = op;
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
