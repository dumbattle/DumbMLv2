using NUnit.Framework;
using DumbML;
using UnityEngine;
using System.Text;

namespace Tests.DumbMLTests {
    public class Training {
        private const string VERY_BASIC_DESCRIPTION =
            "Simple Model - 2 Variables, 1 Add, 1 InputOp, MSE ops - MSE loss - SGD\n" +
            "Output should be continuously decreasing";

        [Test(Description = VERY_BASIC_DESCRIPTION)]
        public void VeryBasicModel() {
            InputOp input = new InputOp(-1, 1);
            Variable b = new Variable(1);
            InputOp truth = new InputOp(-1, 1);

            b.InitValue(() => Random.value);

            var op = new Add(input, b);
            var loss = Loss.MSE(op, truth);
            Model m = new Model(new InputOp[] { input, truth }, new Operation[] { op, loss });
            FloatTensor outputTensor = new FloatTensor(2, 1);
            FloatTensor lossTensor = new FloatTensor(1);

            m.InitTraining(new SGD(momentum: 0, lr: .01f), loss);
            StringBuilder sb = new StringBuilder();

            float prev = float.PositiveInfinity;
            int fail = 0;
            int numIterations = 1000;
            for (int i = 0; i < numIterations; i++) {
                Run();
                float output = lossTensor.data[0];

                if (output > prev) {
                    fail++;
                }
                prev = output;
            }

            m.Dispose();

            if (fail > 0) {
                Debug.Log(sb.ToString());
                Assert.Inconclusive(
                    $"Loss did not decrease for ({fail}) value{(fail > 1 ? "s" : "")} out of {numIterations}" +
                    "This can be caused by the stochastic nature of training.  " +
                    "You should check or rerun to make sure.");

            }

            void Run() {
                m.Call(FloatTensor.FromArray(new[,] { { 1} }), FloatTensor.FromArray(new[,] { { 9 } }));
                m.Backwards();
                m.Call(FloatTensor.FromArray(new[,] { { 1 }, { 3 } }), FloatTensor.FromArray(new[,] { { 9 }, { 11 } })).ToTensors(outputTensor, lossTensor);
                sb.Append($"Output: {outputTensor.data.ContentString()}\nLoss: {lossTensor.data.ContentString()}\n");
                m.Backwards();
            }
        }



        [Test]
        public void MatrixMult() {
            int[] leftShape = { -1, 4, 5 };
            int[] rightShape = { 5, 6 };
            int[] outShape = { -1, 4, 6 };
            int batchCount = 1;
            FloatTensor inputTensor = new FloatTensor(batchCount, 4, 5);
            FloatTensor expectedTensor = new FloatTensor(batchCount, 4, 6);

            for (int i = 0; i < inputTensor.size; i++) {
                inputTensor.data[i] = Random.value;
            }
            for (int i = 0; i < expectedTensor.size; i++) {
                expectedTensor.data[i] = Random.value;
            }

            var inputOp = new InputOp(leftShape);
            var weight = new Variable(rightShape);
            var mm = new MatrixMult(inputOp, weight);

            var expectedOp = new InputOp(outShape);
            var loss = Loss.MSE(mm, expectedOp);

            weight.InitValue(() => Random.Range(-.1f, .1f));


            Model model = new Model(new[] { inputOp, expectedOp }, new Operation[] { mm, loss });

            FloatTensor outputTensor = new FloatTensor(batchCount, 4, 6);
            FloatTensor lossTensor = new FloatTensor(1);
            model.InitTraining(new SGD(), loss);
            StringBuilder sb = new StringBuilder();
             
            float prev = float.PositiveInfinity;
            int fail = 0;
            int numIterations = 1000;
            for (int i = 0; i < numIterations; i++) {
                Run();
            }

            model.Dispose();

            if (fail > 0) {
                Debug.Log(sb.ToString());
                Assert.Inconclusive(
                    $"Loss did not decrease for ({fail}) value{(fail > 1 ? "s" : "")} out of {numIterations}" +
                    "This can be caused by the stochastic nature of training.  " +
                    "You should check or rerun to make sure.");
            }

            void Run() {
                model.Call(inputTensor, expectedTensor).ToTensors(outputTensor, lossTensor);
                model.Backwards();

                float output = lossTensor.data[0];
                string l;
                if (output > prev) {
                    fail++;
                    l = $"<color=red>Loss: {lossTensor.data.ContentString()} </color>\n";
                }
                else {
                    l = $"Loss: {lossTensor.data.ContentString()}\n";
                }
                prev = output;

                sb.Append(l);
            }
        }


    }
}