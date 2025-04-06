using DumbML.RL;
using DumbML.NN;
using DumbML;

namespace FallingRocks {
    public class FRA2C : A2CTrainer {
        Game g;


        public FRA2C(Game g) {
            this.g = g;
            var rlgame = RLGame.Create(
                 (r) => g.ToTensor((FloatTensor)r[0]),
                 (a) => Step(a),
                 () => g.done,
                 () => g.Reset()
             );
            Build(rlgame);

            float Step(Tensor[] actions) {
                var a = ((IntTensor)actions[0])[0];
                var action = (PlayerAction)a;
                var r = g.Update(action);
                return r.reward;
            }
        }

        protected override (int[], DType)[] GetInputSpec() {
            return new[] {
               ( new [] { 1,  g.StateSize()}, DType.Float)
            };
        }
        protected override (int[], DType)[] GetActionSpec() {
            return new[] {
               ( new [] { 1 }, DType.Int)
            };
        }
        public override float DiscoutFactor() {
            return 0.99f;
        }

        protected override Operation[] CreateActor(Operation[] inputs) {
            Operation x = new FullyConnected(256, Activation.ReLU).Build(inputs[0]);
            x = new FullyConnected(256, Activation.ReLU).Build(x);
            x = new FullyConnected(3).Build(x);
            x = DumbML.Math.Softmax(x, -1);

            return new[] { x };
        }
        protected override Operation CreateCritic(Operation[] inputs) {
            Operation x = new FullyConnected(256, Activation.ReLU).Build(inputs[0]);
            x = new FullyConnected(256, Activation.ReLU).Build(x);
            x = new FullyConnected(1).Build(x);
            return x;

        }
        protected override Operation[] ProbsToAction(Operation[] probs) {
            return new[] { new SampleCategorical(probs[0]) };
        }
        protected override Operation CalcLogProbs(Operation[] probs, Operation[] actions) {
            Operation mask = new OneHot(actions[0], 3);        // [bs, 3]
            Operation logProbs = new Log(probs[0]);            // [bs, 3]
            Operation maskedProbs = logProbs * mask;           // [bs, 3]
            Operation result = new ReduceSum(maskedProbs, -1); // [bs]
            return result;
        }
    }

}
