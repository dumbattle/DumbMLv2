using System;
using NUnit.Framework;
using DumbML;


namespace Tests.BLAS.GPU {
    public class TransposeTests : TransposeTestsBase {
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

    public class BroadcastTest : BroadcastTestBase {
        public override void Run(Array src, Array expected) {
            FloatTensor at = FloatTensor.FromArray(src);
            FloatTensor et = FloatTensor.FromArray(expected);
            FloatTensor ot = new FloatTensor(et.shape);

            FloatGPUTensorBuffer input = new FloatGPUTensorBuffer(at.shape);
            FloatGPUTensorBuffer output = new FloatGPUTensorBuffer(et.shape);

            input.CopyFrom(at);

            DumbML.BLAS.GPU.Broadcast.Compute(input, et.shape, output);

            output.CopyTo(ot);
            CollectionAssert.AreEqual(et.data, ot.data, $"E: {et.data.ContentString()}\nG: {ot.data.ContentString()}");

            input.Dispose();
            output.Dispose();
        }
    }
}
