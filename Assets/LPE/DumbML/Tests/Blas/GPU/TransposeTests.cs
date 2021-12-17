using System;
using NUnit.Framework;
using DumbML;


namespace Tests.BLAS {
    namespace GPU {
        public class TransposeTests {
            void Run(Array input, int[] perm, Array expected) {
                FloatTensor it = FloatTensor.FromArray(input);
                FloatTensor et = FloatTensor.FromArray(expected);
                FloatTensor ot = new FloatTensor(et.shape);

                FloatGPUTensorBuffer ib = new FloatGPUTensorBuffer(it.shape);
                FloatGPUTensorBuffer ob = new FloatGPUTensorBuffer(ot.shape);

                ib.CopyFrom(it);
                DumbML.BLAS.GPU.Transpose.Compute(ib, perm, ob);
                ob.CopyTo(ot);

                ib.Dispose();
                ob.Dispose();
                CollectionAssert.AreEqual(et.data, ot.data, ot.data.ContentString());
            }


            [Test]
            public void Test1() {
                float[,,,] input = new float[2,3,4,5];
                int[] perm = { 2, 1, 3, 0 };
                float[,,,] output = new float[4, 3, 5, 2];


                for (int x = 0; x < 2; x++) {
                    for (int y = 0; y < 3; y++) {
                        for (int z = 0; z < 4; z++) {
                            for (int w = 0; w < 5; w++) {
                                float v = UnityEngine.Random.value;
                                input[x, y, z, w] = v;
                                output[z, y, w, x] = v;
                            }
                        }
                    }
                }

                for (int i = 0; i < 100; i++) {
                    Run(input, perm, output);

                }
            }
        }

    }

}