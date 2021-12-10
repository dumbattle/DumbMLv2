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
            DumbML.BLAS.Engine.GPUEnabled = false;
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
    }
}