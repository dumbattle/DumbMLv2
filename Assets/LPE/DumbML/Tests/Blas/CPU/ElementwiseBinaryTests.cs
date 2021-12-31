using NUnit.Framework;
using DumbML;
using System;

namespace Tests.BLAS.CPU {
    public class ElementwiseBinaryTests {
        void Run(Action<FloatCPUTensorBuffer, FloatCPUTensorBuffer, FloatCPUTensorBuffer> compute, Func<float,float,float> singleCompute) {
            RunSimple(compute, singleCompute);
            RunBroadcast(compute, singleCompute);
        }
        void RunSimple(Action<FloatCPUTensorBuffer, FloatCPUTensorBuffer, FloatCPUTensorBuffer> compute, Func<float, float, float> singleCompute) {
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
            CollectionAssert.AreEquivalent(e.buffer, r.buffer, $"Forward Op ERROR.\nExpected: {e.buffer.ContentString()} \nGot: {r.buffer.ContentString()}");

            a.Dispose();
            b.Dispose();
            e.Dispose();
            r.Dispose();
        }
        void RunBroadcast(Action<FloatCPUTensorBuffer, FloatCPUTensorBuffer, FloatCPUTensorBuffer> compute, Func<float, float, float> singleCompute) {
            int[] shapeL = { 5, 1, 5, 5 };
            int[] shapeR = { 5, 1, 5 };
            int[] shapeO = { 5, 5, 5, 5 };

            FloatCPUTensorBuffer a = new FloatCPUTensorBuffer(shapeL);
            FloatCPUTensorBuffer b = new FloatCPUTensorBuffer(shapeR);
            FloatCPUTensorBuffer e = new FloatCPUTensorBuffer(shapeO);

            FloatTensor at = new FloatTensor(shapeL);
            FloatTensor bt = new FloatTensor(shapeR);
            FloatTensor et = new FloatTensor(shapeO);

            for (int x = 0; x < 5; x++) {
                for (int z = 0; z < 5; z++) {
                    for (int w = 0; w < 5; w++) {
                        at[x, 0, z, w] = UnityEngine.Random.Range(-1, 1);
                    }
                }
            }
            for (int y = 0; y < 5; y++) {
                for (int w = 0; w < 5; w++) {
                    bt[y, 0, w] = UnityEngine.Random.Range(-1, 1);
                }
            }
            for (int x = 0; x < 5; x++) {
                for (int y = 0; y < 5; y++) {
                    for (int z = 0; z < 5; z++) {
                        for (int w = 0; w < 5; w++) {

                            et[x,y,z,w] = singleCompute(at[x, 0, z, w], bt[y, 0, w]);
                        }
                    }
                }
            }
            a.CopyFrom(at);
            b.CopyFrom(bt);
            e.CopyFrom(et);

            FloatCPUTensorBuffer r = new FloatCPUTensorBuffer(shapeO);
            compute(a, b, r);
            CollectionAssert.AreEquivalent(e.buffer, r.buffer, $"Forward Op ERROR.\nExpected: {e.buffer.ContentString()} \nGot: {r.buffer.ContentString()}");

            a.Dispose();
            b.Dispose();
            e.Dispose();
            r.Dispose();
        }



        [Test]
        public void Add() {
            Run((a, b, r) => DumbML.BLAS.CPU.ElementwiseBinary.Add(a, b, r), (a, b) => a + b);
        }
        [Test]
        public void Multiply() {
            Run((a, b, r) => DumbML.BLAS.CPU.ElementwiseBinary.Multiply(a, b, r), (a, b) => a * b);
        }
        [Test]
        public void Subtract() {
            Run((a, b, r) => DumbML.BLAS.CPU.ElementwiseBinary.Subtract(a, b, r), (a, b) => a - b);
        }
    }
}
