using NUnit.Framework;
using DumbML;
using System;

namespace Tests.BLAS.CPU {
    public class ElementwiseBinaryTests {
        void Run(Action<CPUTensorBuffer, CPUTensorBuffer, CPUTensorBuffer> compute, Func<float,float,float> singleCompute) {
            int[] shape = { 5, 5, 5, 5 };

            CPUTensorBuffer a = new CPUTensorBuffer(shape);
            CPUTensorBuffer b = new CPUTensorBuffer(shape);
            CPUTensorBuffer e = new CPUTensorBuffer(shape);

            Tensor at = new Tensor(shape);
            Tensor bt = new Tensor(shape);
            Tensor et = new Tensor(shape);

            for (int i = 0; i < at.size; i++) {
                at.data[i] = UnityEngine.Random.Range(-1, 1);
                bt.data[i] = UnityEngine.Random.Range(-1, 1);
                et.data[i] = singleCompute(at.data[i], bt.data[i]);
            }
            a.CopyFrom(at);
            b.CopyFrom(bt);
            e.CopyFrom(et);

            CPUTensorBuffer r = new CPUTensorBuffer(a.shape);
            compute(a, b, r);
            CollectionAssert.AreEqual(e.buffer, r.buffer, $"Forward Op ERROR.\nExpected: {e.buffer.ContentString()} \nGot: {r.buffer.ContentString()}");
        }


        [Test]
        public void Add() {
            Run((a, b, r) => DumbML.BLAS.CPU.ElementwiseBinary.Add(a, b, r), (a, b) => a + b);
        }
    }
}
