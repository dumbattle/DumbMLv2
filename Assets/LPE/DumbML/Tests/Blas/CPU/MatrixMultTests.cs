using NUnit.Framework;
using DumbML;
using System;

namespace Tests.BLAS.CPU {
    public class MatrixMultTests {
        void Run(Array left, Array right, Array expected, bool tl, bool tr) {
            FloatTensor a = FloatTensor.FromArray(left);
            FloatTensor b = FloatTensor.FromArray(right);
            FloatTensor e = FloatTensor.FromArray(expected);
            FloatTensor o = new FloatTensor(e.shape);

            FloatCPUTensorBuffer ab = new FloatCPUTensorBuffer(a.shape);
            FloatCPUTensorBuffer bb = new FloatCPUTensorBuffer(b.shape);
            FloatCPUTensorBuffer ob = new FloatCPUTensorBuffer(o.shape);

            ab.CopyFrom(a);
            bb.CopyFrom(b);
       
            try {
                DumbML.BLAS.CPU.MatrixMult.Compute(ab, bb, ob, tl, tr);
                ob.CopyTo(o);
                CollectionAssert.AreEqual(e.data, o.data);
            }
            finally {
                ab.Dispose();
                bb.Dispose();
                ob.Dispose();
            }
        }



        [Test(Description = "Shapes: (1,3)x(3,4)")]
        public void Test1() {
            float[,] a = { { 1, 2, 3 } };
            float[,] b = { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } };
            float[,] e = { { 38, 44, 50, 56 } };
            Run(a, b, e, false, false);
        }

        [Test(Description = "Shapes: (3)x(3,4) - (3) is invalid. Expand to (1,3) to do MatrixMult")]
        public void Test2() {
            float[] a = { 1, 2, 3 };
            float[,] b = { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } };
            float[,] e = { { 38, 44, 50, 56 } };
            Assert.Throws<ArgumentException>(() => Run(a, b, e, false, false));
        }

        [Test(Description = "Shapes: (2,2,2,2)x(2,2,3)")]
        public void Test3() {
            float[,,,] _a = { { { { 1, 2 }, { 3, 4 } }, { { 5, 6 }, { 7, 8 } } }, { { { -1, -2 }, { -3, -4 } }, { { -5, -6 }, { -7, -8 } } } };
            float[,,] _b = { { { 1, 2, 3 }, { 4, 5, 6 } }, { { -1, 2, 3 }, { -4, 5, -6 } } };
            float[,,,] _e =
                { { { { 9, 12, 15 },
                      { 19, 26, 33 } },
                    { { -29, 40, -21 },
                      { -39, 54, -27 } } },
                  { { { -9, -12, -15 },
                      { -19, -26, -33 } },
                    { { 29, -40, 21 },
                      { 39, -54, 27 } } } };
            Run(_a, _b, _e, false, false);
        }
        
        [Test(Description = "Shapes: (3,1)Tx(3,4)")]
        public void Test4() {
            float[,] a = { { 1 }, { 2 }, {3 } };
            float[,] b = { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } };
            float[,] e = { { 38, 44, 50, 56 } };
            Run(a, b, e, true, false);
        }

        [Test(Description = "Shapes: (1,3)x(4,3)T")]
        public void Test5() {
            float[,] a = { { 1, 2, 3 } };
            float[,] b = { { 1, 5, 9 }, { 2, 6, 10 }, { 3, 7, 11 }, { 4, 8, 12 } };
            float[,] e = { { 38, 44, 50, 56 } };
            Run(a, b, e, false, true);
        }

        [Test(Description = "Shapes: (3,1)Tx(4,3)T")]
        public void Test6() {
            float[,] a = { { 1 }, { 2 }, { 3 } };
            float[,] b = { { 1, 5, 9 }, { 2, 6, 10 }, { 3, 7, 11 }, { 4, 8, 12 } };
            float[,] e = { { 38, 44, 50, 56 } };
            Run(a, b, e, true, true);
        }


    }
}
