using System;
using NUnit.Framework;


namespace Tests.BLAS {
    public abstract class TransposeTestsBase {

        protected abstract void Run(Array input, int[] perm, Array expected);


        [Test(Description ="Shape: [2, 3, 4, 5]\nPerm:   [2, 1, 3, 0]")]
        public void _2_3_4_5__2_1_3_0_() {
            float[,,,] input = new float[2, 3, 4, 5];
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