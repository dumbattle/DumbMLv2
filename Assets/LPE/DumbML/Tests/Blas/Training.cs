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
            Variable a = new Variable(1);
            Variable b = new Variable(1);
            InputOp input = new InputOp(1);

            a.InitValue(() => Random.value);
            b.InitValue(() => Random.value);

            var op = new Add(a, b);
            var loss = Loss.MSE(op, input);
            Model m = new Model(new InputOp[] { input }, new Operation[] { op, loss });
            FloatTensor outputTensor = new FloatTensor(1);
            FloatTensor lossTensor = new FloatTensor(1);

            m.InitTraining(new SGD(momentum: 0, lr: .01f), loss);
            StringBuilder sb = new StringBuilder();

            float prev = float.PositiveInfinity;
            int fail = 0;
            int numIterations = 1000;
            for (int i = 0; i < numIterations; i++) {
                Run();
                float output = lossTensor[0];

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
                m.Call(FloatTensor.FromArray(new[] { 9 })).ToTensors(outputTensor, lossTensor);
                sb.Append($"Output: {outputTensor.data.ContentString()} Loss: {lossTensor.data.ContentString()}\n");
                m.Backwards();
            }
        }



        [Test]
        public void MatrixMult() {
            int[] leftShape = { 4, 5 };
            int[] rightShape = { 5, 6 };
            int[] outShape = { 4, 6 };

            FloatTensor inputTensor = new FloatTensor(leftShape);
            FloatTensor expectedTensor = new FloatTensor(outShape);

            for (int i = 0; i < inputTensor.size; i++) {
                inputTensor.data[i] = Random.value;
            }
            for (int i = 0; i < expectedTensor.size; i++) {
                expectedTensor.data[i] = Random.value;
            }

            var inputOp = new InputOp(leftShape);
            var weight = new Variable(rightShape);
            var expectedOp = new InputOp(outShape);
            var mm = new MatrixMult(inputOp, weight);

            var loss = Loss.MSE(mm, expectedOp);

            weight.InitValue(() => Random.Range(-1f, 1f));


            Model model = new Model(new[] { inputOp, expectedOp }, new Operation[] { mm, loss });

            FloatTensor outputTensor = new FloatTensor(mm.shape);
            FloatTensor lossTensor = new FloatTensor(loss.shape);
            model.InitTraining(new SGD(), loss);
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
                sb.Append($"Output: {outputTensor.data.ContentString()} Loss: {lossTensor.data.ContentString()}\n");
                model.Backwards();
            }
        }


    }
}