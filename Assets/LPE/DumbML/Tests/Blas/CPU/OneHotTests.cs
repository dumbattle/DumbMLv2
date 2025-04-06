using NUnit.Framework;
using DumbML;
using System;



namespace Tests.BLAS.CPU {
    public class OneHotTests {
        [Test]
        public void Run() {
            Tensor<int> indices = IntTensor.FromArray(new int[,] { { 0, 1, 2 }, { 3, 0, 1 }, { 2, 2, 1 } });
            Tensor<int> target = IntTensor.FromArray(
                new int[,,] { { { 1, 0, 0, 0 }, { 0, 1, 0, 0 }, { 0, 0, 1, 0 } }, { { 0, 0, 0, 1 }, { 1, 0, 0, 0 }, { 0, 1, 0, 0 } }, { { 0, 0, 1, 0 }, { 0, 0, 1, 0 }, { 0, 1, 0, 0 } } });

            CPUTensorBuffer<int> indBuffer = new IntCPUTensorBuffer(indices.shape);
            CPUTensorBuffer<float> outBuffer = new FloatCPUTensorBuffer(target.shape);
            indBuffer.CopyFrom(indices);

            DumbML.BLAS.CPU.OneHot.Compute(indBuffer, 4, 1, 0, outBuffer);



            CollectionAssert.AreEqual(target.data, outBuffer.buffer, outBuffer.buffer.ContentString());

            indBuffer.Dispose();
            outBuffer.Dispose();
        }
    }
}
