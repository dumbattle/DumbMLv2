using System;
using NUnit.Framework;


namespace Tests.BLAS {
    public abstract class BroadcastTestBase {
        public abstract void Run(Array src, Array expected);
        [Test]
        public void Test1() {
            int[] inputShape = { 3, 4 };
            int[] targetShape = { 4, 7, 3, 4 };

            float[,] a = new float[3, 4];
            float[,,,] b = new float[4, 7, 3, 4];

            for (int a1 = 0; a1 < inputShape[0]; a1++) {
                for (int a2 = 0; a2 < inputShape[1]; a2++) {
                    float v = UnityEngine.Random.Range(0, 5);

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