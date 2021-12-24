using System;
using NUnit.Framework;
using DumbML;


namespace Tests.BLAS {
    namespace GPU {
        public class TransposeTests : TransposeTestsBase  {
            protected override void Run(Array input, int[] perm, Array expected) {
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


        }

    }
}