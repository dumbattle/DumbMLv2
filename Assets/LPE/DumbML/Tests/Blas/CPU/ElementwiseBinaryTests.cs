using NUnit.Framework;
using DumbML;
using System;

namespace Tests.BLAS.CPU {
    public class ElementwiseBinaryTests {
        void Run(Action<FloatCPUTensorBuffer, FloatCPUTensorBuffer, FloatCPUTensorBuffer> compute, Func<float,float,float> singleCompute) {
            int[] shape = { 5, 5, 5, 5 };

            FloatCPUTensorBuffer a = new FloatCPUTensorBuffer(shape);
            FloatCPUTensorBuffer b = new FloatCPUTensorBuffer(shape);
            FloatCPUTensorBuffer e = new FloatCPUTensorBuffer(shape);

            FloatTensor at = new FloatTensor(shape);
            FloatTensor bt = new FloatTensor(shape);
            FloatTensor et = new FloatTensor(shape);

            for (int i = 0; i < at.size; i++) {
                at.data[i] = UnityEngine.Random.Range(-1, 1);
                bt.data[i] = UnityEngine.Random.Range(-1, 1);
                et.data[i] = singleCompute(at.data[i], bt.data[i]);
            }
            a.CopyFrom(at);
            b.CopyFrom(bt);
            e.CopyFrom(et);

            FloatCPUTensorBuffer r = new FloatCPUTensorBuffer(a.shape);
            compute(a, b, r);
            CollectionAssert.AreEqual(e.buffer, r.buffer, $"Forward Op ERROR.\nExpected: {e.buffer.ContentString()} \nGot: {r.buffer.ContentString()}");
        }


        [Test]
        public void Add() {
            Run((a, b, r) => DumbML.BLAS.CPU.ElementwiseBinary.Add(a, b, r), (a, b) => a + b);
        }
    }
}
