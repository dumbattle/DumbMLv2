using NUnit.Framework;
using DumbML;
using System;

namespace Tests.BLAS.CPU {
    public class MatrixMultTests {
        void Run(Array left, Array right, Array expected) {
            Tensor a = Tensor.FromArray(left);
            Tensor b = Tensor.FromArray(right);
            Tensor e = Tensor.FromArray(expected);
            Tensor o = new Tensor(e.shape);

            CPUTensorBuffer ab = new CPUTensorBuffer(a.shape);
            CPUTensorBuffer bb = new CPUTensorBuffer(b.shape);
            CPUTensorBuffer ob = new CPUTensorBuffer(o.shape);

            ab.CopyFrom(a);
            bb.CopyFrom(b);


            DumbML.BLAS.CPU.MatrixMult.Compute(ab, bb, ob);
            ob.CopyTo(o);

            CollectionAssert.AreEqual(e.data, o.data);
        }



        [Test(Description = "Shapes: (1,3)x(3,4)")]
        public void Test1() {
            float[,] a = { { 1, 2, 3 } };
            float[,] b = { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } };
            float[,] e = { { 38, 44, 50, 56 } };
            Run(a, b, e);
        }



        [Test(Description = "Shapes: (3)x(3,4) - (3) is invalid. Expand to (1,3) to do MatrixMult")]
        public void Test2() {
            float[] a = { 1, 2, 3 };
            float[,] b = { { 1, 2, 3, 4 }, { 5, 6, 7, 8 }, { 9, 10, 11, 12 } };
            float[,] e = { { 38, 44, 50, 56 } };
            Assert.Throws<ArgumentException>(() => Run(a, b, e));
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
            Run(_a, _b, _e);
        }
   
    }
}
