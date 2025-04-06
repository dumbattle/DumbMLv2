using UnityEngine;
using DumbML.RL;
using DumbML.NN;
using DumbML;


namespace FreeSnake {
    public class FreeSnakeMain : MonoBehaviour {
        [SerializeField] GameParameters parameters;
        [Space]
        [SerializeField] GameObject playerObject;
        [SerializeField] GameObject targetObject;

        Game g;
        A2CTrainer trainer;
        System.Diagnostics.Stopwatch sw = new System.Diagnostics.Stopwatch();
        [TextArea]
        public string debugString;

        public int speed = 100;
        void Start() {
            DumbML.BLAS.Engine.GPUEnabled = false;
            g = new Game(parameters);
            g.Reset();
            playerObject.transform.localScale
                = new Vector3(parameters.playerRadius, parameters.playerRadius, 1) * 2;
            targetObject.transform.localScale
                = new Vector3(parameters.targetRadius, parameters.targetRadius, 1) * 2;
            trainer = new FreeSnakeA2C(g);
        }

        void Update() {
            sw.Reset();
            sw.Start();
            for (int i = 0; i < speed; i++) {
                trainer.Step();
                if (sw.ElapsedMilliseconds > 1000) {
                    break;
                }
            }

            DrawGame();
            debugString = trainer.DEBUG_STRING;
        }

        private void OnDestroy() {
            trainer.Dispose();
        }

        void DrawGame() {
            playerObject.transform.position = g.playerPosition;
            targetObject.transform.position = g.targetPosition;
        }
    }

    public class FreeSnakeA2C : A2CTrainer {
        Game g;
        static Vector2[] moveOptions = { Vector2.up, Vector2.left, Vector2.right, Vector2.down };

        public FreeSnakeA2C(Game g) {
            this.g = g;
            var rlgame = RLGame.Create(
                 (r) => g.ToTensor((FloatTensor)r[0]),
                 (a) => Step(a),
                 () => g.done,
                 () => g.Reset()
             );
            Build(rlgame);

            float Step(Tensor[] actions) {
                var a = ((IntTensor)actions[0])[0, 0];
                var action = moveOptions[a];
                var r = g.Update(action);
                return r;
            }
        }

        protected override (int[], DType)[] GetInputSpec() {
            return new[] {
                (new [] { 2 }, DType.Float)
            };
        }
        protected override (int[], DType)[] GetActionSpec() {
            return new[] {
                (new [] { 1 }, DType.Int)
            };
        }
        public override float DiscoutFactor() {
            return 0.5f;
        }

        protected override Operation[] CreateActor(Operation[] inputs) {
            Operation x;
            x = new FullyConnected(32, Activation.ReLU).Build(inputs[0]);
            x = new FullyConnected(4).Build(x);
            x = DumbML.Math.Softmax(x, -1);
            return new[] { x };
        }
        protected override Operation CreateCritic(Operation[] inputs) {
            Operation x;
            x = new FullyConnected(32, Activation.ReLU).Build(inputs[0]);
            x = new FullyConnected(1).Build(x);
            return x;

        }
        protected override Operation[] ProbsToAction(Operation[] probs) {
            return new[] {  new SampleCategorical(probs[0]) };
        }
        protected override Operation CalcLogProbs(Operation[] probs, Operation[] actions) {
            Operation mask = new OneHot(actions[0], 4);        // [bs, 1, 4]
            mask = new Reshape(mask, mask.shape[0], 4);        // [bs, 4]
            Operation logProbs = new Log(probs[0]);            // [bs, 4]
            Operation maskedProbs = logProbs * mask;           // [bs, 4]
            Operation result = new ReduceSum(maskedProbs, -1); // [bs]
            result = new AddDims(result, new[] { 1 });
            return result;
        }
    }
}