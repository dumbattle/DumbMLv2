using NUnit.Framework;
using DumbML;



namespace Tests.BLAS.CPU {
    public class SetValuesTest {
        void Run(int[] shape, float val) {
            FloatTensor t = new FloatTensor(shape);

            FloatCPUTensorBuffer tb = new FloatCPUTensorBuffer(t.shape);
            tb.CopyFrom(t);

            DumbML.BLAS.CPU.SetValues.Run(tb, val);

            foreach (var v in tb.buffer) {
                Assert.AreEqual(v, val);
            }
        }
        [Test]
        public void One() {
            int[] shape = { 3, 4, 5 };
            Run(shape, 1);
        }
        [Test]
        public void Value() {
            int[] shape = { 3, 4, 5 };
            Run(shape, 1.3f);
        }
        [Test]
        public void Zero() {
            int[] shape = { 3, 4, 5 };
            Run(shape, 0);
        }
    }
}
