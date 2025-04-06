using System.Collections.Generic;
using System.Linq;
using DumbML.NN;
using LPE;
using UnityEngine;

namespace DumbML.RL {
    public abstract class A2CTrainer {
        public Model actor;
        RLGame game;

        Model stepModel;
        Model forwardModel;


        public string DEBUG_STRING;
        Tensor TEST_TENSOR;
        protected Operation TEST_OP;
        //***************************************************************************************************************
        //Training Variables
        //***************************************************************************************************************
        Tensor[] _currentState;
        Tensor[] _currentStateBatched;
        Tensor[] _currentActions;
        ObjectPool<A2CExperience> xpPool;
        List<A2CExperience> trajectory = new List<A2CExperience>();
        FloatTensor _rewardTensor = new FloatTensor(1);

        Tensor[] _fowardModelInputs;
        Tensor[] tsbsDest;
        TensorStackBuilder[] tsbs;
        int batch_size = 64;
        //***************************************************************************************************************
        //Build
        //***************************************************************************************************************

        protected void Build(RLGame game) {
            this.game = game;

            InitTrainingVariables();
            BuildModels();
        }
        
        private void InitTrainingVariables() {
            var inputSpec = GetInputSpec();
            var actionSpec = GetActionSpec();

            //_currentState = (from x in inputSpec select Tensor.Get(x.Item2, x.Item1)).ToArray();

            _currentActions = new Tensor[inputSpec.Length];
            _currentState = new Tensor[inputSpec.Length];
            _currentStateBatched = new Tensor[inputSpec.Length];

            for (int i = 0; i < inputSpec.Length; i++) {
                int[] s = new int[inputSpec[i].Item1.Length + 1];

                for (int j = 0; j < inputSpec[i].Item1.Length; j++) {
                    s[j + 1] = inputSpec[i].Item1[j];
                }
                s[0] = 1;
                _currentState[i] =        Tensor.Get(inputSpec[i].Item2, inputSpec[i].Item1);
                _currentStateBatched[i] = Tensor.Get(inputSpec[i].Item2, s);
            }
            
            for (int i = 0; i < actionSpec.Length; i++) {
                int[] s = new int[actionSpec[i].Item1.Length + 1];

                for (int j = 0; j < actionSpec[i].Item1.Length; j++) {
                    s[j + 1] = actionSpec[i].Item1[j];
                }
                s[0] = 1;
                _currentActions[i] = Tensor.Get(actionSpec[i].Item2, s);
            }

            xpPool = new ObjectPool<A2CExperience>(() => {
                var result = new A2CExperience();
                result.state = (from x in inputSpec select Tensor.Get(x.Item2, x.Item1)).ToArray();
                result.action = (from x in actionSpec select Tensor.Get(x.Item2, x.Item1)).ToArray();
                return result;
            });

            _fowardModelInputs =
                (from x in inputSpec select Tensor.Get(x.Item2, x.Item1))
                .Concat((from x in actionSpec select Tensor.Get(x.Item2, x.Item1))
                .Concat(new[] { _rewardTensor })).ToArray();
            tsbs =
                (from x in inputSpec select TensorStackBuilder.Get(x.Item2, x.Item1))
                .Concat((from x in actionSpec select TensorStackBuilder.Get(x.Item2, x.Item1))
                .Concat(new[] { new TensorStackBuilder<float>(new[] { 1 }) })).ToArray();

            tsbsDest =
                (from x in inputSpec select GetDestTensor(x.Item2, x.Item1))
                .Concat((from x in actionSpec select GetDestTensor(x.Item2, x.Item1))
                .Concat(new[] { GetDestTensor(DType.Float, new[] { 1 }) })).ToArray();
            
            for (int i = 0; i < tsbsDest.Length; i++) {
                tsbs[i].Begin(tsbsDest[i]);
            }

            Tensor GetDestTensor(DType dtype, int[] shape) {
                var s = new List<int>() { batch_size };
                s.AddRange(shape);
                return Tensor.Get(dtype, s.ToArray());
            }
        }

        private void BuildModels() {
            // create inputs - Add in batch dimension
            InputOp[] inputOps = new InputOp[_currentState.Length];

            for (int i = 0; i < _currentState.Length; i++) {
                int[] s = new int[_currentState[i].shape.Length + 1];

                for (int j = 0; j < s.Length - 1; j++) {
                    s[j + 1] = _currentState[i].shape[j];
                }
                s[0] = batch_size;
                inputOps[i] = new InputOp(_currentState[i].dtype, s);
            }

            // create component models
            var probsOp = CreateActor(inputOps);
            var criticOp = CreateCritic(inputOps);
            TEST_OP = probsOp[0];
            TEST_TENSOR = Tensor.Get(TEST_OP.dtype, new[] { 1, 4 });
            var actionOp = ProbsToAction(probsOp);

            // Calculate log probs
            InputOp[] actionInputs = new InputOp[_currentActions.Length];
            for (int i = 0; i < _currentActions.Length; i++) {
                actionInputs[i] = new InputOp(_currentActions[i].dtype, _currentActions[i].shape);
            }
            var logProbOp = CalcLogProbs(probsOp, actionInputs);
            // compute loss
            InputOp rewardInput = new InputOp(batch_size, 1);
            Operation adv = rewardInput - criticOp;
            Operation aloss = -new ReduceSum(logProbOp * new NoGrad(adv));
            Operation closs = new ReduceSum(new Square(adv));
            var loss = aloss + closs;

            actor = new Model(inputOps, probsOp);
            stepModel = new Model(inputOps, actionOp[0]);

            forwardModel = new Model(inputOps.Concat(actionInputs).Concat(new[] { rewardInput }).ToArray(), loss);
            forwardModel.InitTraining(GetOptimizer(), loss);
        }

        //***************************************************************************************************************
        //Training Methods
        //***************************************************************************************************************

        public void Step() {
            // Forward 1 step
            game.GetState(_currentState);

            for (int i = 0; i < _currentState.Length; i++) {
                _currentStateBatched[i].CopyFrom(_currentState[i], true);
            }
            stepModel.Call(_currentStateBatched).ToTensors(_currentActions);
          
            new[] { TEST_OP.outputBuffer }.ToTensors(TEST_TENSOR);
            DEBUG_STRING = TEST_TENSOR.ToString();

            float reward = game.Step(_currentActions);

            // record experience
            var xp = xpPool.Get();

            for (int i = 0; i < _currentState.Length; i++) {
                xp.state[i].CopyFrom(_currentState[i]);
            }
            for (int i = 0; i < _currentActions.Length; i++) {
                xp.action[i].CopyFrom(_currentActions[i], true);
            }
            xp.reward = reward;
            trajectory.Add(xp);

            // end game
            if (game.IsDone()) {
                game.Reset();
                EndTrajectory();
                Train();
                ClearTrajectory();
            }
        }

        void EndTrajectory() {
            float r = 0;
            for (int i = trajectory.Count - 1; i >= 0; i--) {
                r *= DiscoutFactor();
                r += trajectory[i].reward;
                trajectory[i].reward = r;
            }
        }

        void Train() {
            for (int step = trajectory.Count - 1; step >= 0; step--) {
                A2CExperience xp = trajectory[step];
                for (int i = 0; i < xp.state.Length; i++) {
                    _fowardModelInputs[i].CopyFrom(xp.state[i]);
                }
                for (int i = 0; i < xp.action.Length; i++) {
                    _fowardModelInputs[i + xp.state.Length].CopyFrom(xp.action[i]);
                }
                _rewardTensor[0] = xp.reward;

                bool done = false;
                for (int i = 0; i < tsbs.Length; i++) {
                    tsbs[i].Add(_fowardModelInputs[i]);
                    done = tsbs[i].Done();
                }

                if (done) {
                    forwardModel.Call(tsbsDest);
                    forwardModel.Backwards();

                    for (int j = 0; j < tsbs.Length; j++) {
                        tsbs[j].Begin(tsbsDest[j]);
                    }
                }
            }
        }

        void ClearTrajectory() {
            foreach (var xp in trajectory) {
                xpPool.Return(xp);
            }
            trajectory.Clear();
        }

        public void Dispose() {
            actor.Dispose();
            stepModel.Dispose();
            forwardModel.Dispose(true);
        }

        //***************************************************************************************************************
        //Hyperparameters
        //***************************************************************************************************************

        public virtual float DiscoutFactor() {
            return .9f;
        }
        public virtual Optimizer GetOptimizer() {
            return new SGD(0.0001f, 0);
        }

        //***************************************************************************************************************
        //Abstract Methods
        //***************************************************************************************************************

        protected abstract (int[], DType)[] GetInputSpec();
        protected abstract (int[], DType)[] GetActionSpec();

        protected abstract Operation[] CreateActor(Operation[] inputs);
        protected abstract Operation CreateCritic(Operation[] inputs);

        protected abstract Operation[] ProbsToAction(Operation[] probs);
        protected abstract Operation CalcLogProbs(Operation[] probs, Operation[] actions);
    }
}
