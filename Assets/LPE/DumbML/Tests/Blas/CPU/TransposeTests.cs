using NUnit.Framework;
using DumbML;
using System;



namespace Tests.BLAS.CPU {
    public class TransposeTests : TransposeTestsBase {
        protected override void Run(Array input, int[] perm, Array expected) {
            FloatTensor it = FloatTensor.FromArray(input);
            FloatTensor et = FloatTensor.FromArray(expected);
            FloatTensor ot = new FloatTensor(et.shape);

            FloatCPUTensorBuffer ib = new FloatCPUTensorBuffer(it.shape);
            FloatCPUTensorBuffer ob = new FloatCPUTensorBuffer(ot.shape);

            ib.CopyFrom(it);
            DumbML.BLAS.CPU.Transpose.Compute(ib, perm, ob);
            ob.CopyTo(ot);

            ib.Dispose();
            ob.Dispose();
            CollectionAssert.AreEqual(et.data, ot.data, ot.data.ContentString());
        }


    }

}
