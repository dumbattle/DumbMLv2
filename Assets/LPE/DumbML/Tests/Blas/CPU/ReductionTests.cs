using NUnit.Framework;
using DumbML;
using System;



namespace Tests.BLAS.CPU {
    public class ReductionTests {
        void Run(Array src, int[] axis, Array expected) {
            FloatTensor at = FloatTensor.FromArray(src);
            FloatTensor et = FloatTensor.FromArray(expected);
            FloatCPUTensorBuffer input = new FloatCPUTensorBuffer(at.shape);
            FloatCPUTensorBuffer output = new FloatCPUTensorBuffer(et.shape);
            input.CopyFrom(at);
            DumbML.BLAS.CPU.Reduction.Sum(input, axis, output);
            CollectionAssert.AreEqual(et.data, output.buffer);
        }
        [Test]
        public void Sum1() {
            float[,,] a =
                { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } },
                  { { 10, 11, 12 }, { 13, 14, 15 }, { 16, 17, 18 } },
                  { { 19, 20, 21 }, { 22, 23, 24 }, { 25, 26, 27 } } };

            int[] reduction = { 0 };
            float[,] e = { { 30, 33, 36 }, { 39, 42, 45 }, { 48, 51, 54 } };

            Run(a, reduction, e);
        }
        [Test]
        public void Sum2() {
            float[,,] a =
                { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } },
                  { { 10, 11, 12 }, { 13, 14, 15 }, { 16, 17, 18 } },
                  { { 19, 20, 21 }, { 22, 23, 24 }, { 25, 26, 27 } } };

            int[] reduction = { 1 };
            float[,] e = { { 12, 15, 18 }, { 39, 42, 45 }, { 66, 69, 72 } };

            Run(a, reduction, e);
        }
        [Test]
        public void Sum3() {
            float[,,] a =
                { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } },
                  { { 10, 11, 12 }, { 13, 14, 15 }, { 16, 17, 18 } },
                  { { 19, 20, 21 }, { 22, 23, 24 }, { 25, 26, 27 } } };

            int[] reduction = { 2 };
            float[,] e = { { 6, 15, 24 }, { 33, 42, 51 }, { 60, 69, 78 } };

            Run(a, reduction, e);
        }
        [Test]
        public void Sum4() {
            float[,,] a =
                { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } },
                  { { 10, 11, 12 }, { 13, 14, 15 }, { 16, 17, 18 } },
                  { { 19, 20, 21 }, { 22, 23, 24 }, { 25, 26, 27 } } };

            int[] reduction = { 0, 1 };
            float[] e = { 117, 126, 135 };

            Run(a, reduction, e);
        }
        [Test]
        public void Sum5() {
            float[,,] a =
                { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } },
                  { { 10, 11, 12 }, { 13, 14, 15 }, { 16, 17, 18 } },
                  { { 19, 20, 21 }, { 22, 23, 24 }, { 25, 26, 27 } } };

            int[] reduction = { 0, 2 };
            float[] e = { 99, 126, 153 };

            Run(a, reduction, e);
        }
        [Test]
        public void Sum6() {
            float[,,] a =
                { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } },
                  { { 10, 11, 12 }, { 13, 14, 15 }, { 16, 17, 18 } },
                  { { 19, 20, 21 }, { 22, 23, 24 }, { 25, 26, 27 } } };

            int[] reduction = { 1, 2 };
            float[] e = { 45, 126, 207 };

            Run(a, reduction, e);
        }
        [Test]
        public void Sum7() {
            float[,,] a =
                { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } },
                  { { 10, 11, 12 }, { 13, 14, 15 }, { 16, 17, 18 } },
                  { { 19, 20, 21 }, { 22, 23, 24 }, { 25, 26, 27 } } };

            int[] reduction = { 0, 1, 2 };
            float[] e = { 378 };

            Run(a, reduction, e);
        }

    }


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
