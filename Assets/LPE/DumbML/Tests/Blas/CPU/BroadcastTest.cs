using NUnit.Framework;
using DumbML;
using System;



namespace Tests.BLAS.CPU {
    public class BroadcastTest : BroadcastTestBase {
        public override void Run(Array src, Array expected) {
            FloatTensor at = FloatTensor.FromArray(src);
            FloatTensor et = FloatTensor.FromArray(expected);

            FloatCPUTensorBuffer input = new FloatCPUTensorBuffer(at.shape);
            FloatCPUTensorBuffer output = new FloatCPUTensorBuffer(et.shape);

            input.CopyFrom(at);

            DumbML.BLAS.CPU.Broadcast.Compute(input, et.shape, output);
            CollectionAssert.AreEqual(et.data, output.buffer, $"E: {et.data.ContentString()}\nG: {output.buffer.ContentString()}");

            input.Dispose();
            output.Dispose();
        }
    }
}
