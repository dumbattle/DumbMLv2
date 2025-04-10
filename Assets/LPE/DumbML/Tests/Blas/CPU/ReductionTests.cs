﻿using NUnit.Framework;
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

            input.Dispose();
            output.Dispose();   
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
        [Test]
        public void Sum8() {
            float[,,] a =
                { { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } },
                  { { 10, 11, 12 }, { 13, 14, 15 }, { 16, 17, 18 } },
                  { { 19, 20, 21 }, { 22, 23, 24 }, { 25, 26, 27 } } };

            int[] reduction = null;
            float[] e = { 378 };

            Run(a, reduction, e);
        }

        [Test]
        public void Sum9() {
            float[,,] a = new float[5, 5, 4];
            float sum = 0;

            for (int x = 0; x < a.GetLength(0); x++) {
                for (int y = 0; y < a.GetLength(1); y++) {
                    for (int z = 0; z < a.GetLength(2); z++) {
                        a[x, y, z] = UnityEngine.Random.value;
                        sum += a[x, y, z];
                    }
                }
            }
            int[] reduction = { 0, 1, 2 };
            float[] e = { sum };

            Run(a, reduction, e);
        }
        [Test]
        public void Sum10() {
            float[,,] a = new float[4, 18, 18];
            float[,] e = new float[a.GetLength(0), a.GetLength(2)];

            for (int x = 0; x < a.GetLength(0); x++) {
                for (int y = 0; y < a.GetLength(1); y++) {
                    for (int z = 0; z < a.GetLength(2); z++) {
                        a[x, y, z] = UnityEngine.Random.value;
                        e[x, z] += a[x, y, z];
                    }
                }
            }
            int[] reduction = { 1 };

            Run(a, reduction, e);
        }
    }
}
