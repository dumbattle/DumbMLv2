using NUnit.Framework;
using DumbML;
using System;



namespace Tests.BLAS.CPU {
    public class BroadcastTest {
        static void Run(Array src, Array expected) {
            FloatTensor at = FloatTensor.FromArray(src);
            FloatTensor et = FloatTensor.FromArray(expected);

            FloatCPUTensorBuffer input = new FloatCPUTensorBuffer(at.shape);
            FloatCPUTensorBuffer output = new FloatCPUTensorBuffer(et.shape);

            input.CopyFrom(at);

            DumbML.BLAS.CPU.Broadcast.Compute(input, et.shape, output);
            CollectionAssert.AreEqual(et.data, output.buffer, $"E: {et.data.ContentString()}\nG: {output.buffer.ContentString()}");
        }

        [Test]
        public void Test1() {
            int[] inputShape = { 3, 4 };
            int[] targetShape = { 4, 7, 3, 4 };

            float[,] a = new float[3,4];
            float[,,,] b= new float[4, 7, 3, 4];

            for (int a1 = 0; a1 < inputShape[0]; a1++) {
                for (int a2 = 0; a2 < inputShape[1]; a2++) {
                    float v = UnityEngine.Random.Range(0,5);

                    a[a1, a2] = v;
                    for (int b1 = 0; b1 < targetShape[0]; b1++) {
                        for (int b2 = 0; b2 < targetShape[1]; b2++) {
                            b[b1, b2, a1, a2] = v;
                        }
                    }
                }
            }

            Run(a, b);
        }
    }
}
