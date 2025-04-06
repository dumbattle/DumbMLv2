using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using DumbML;
using System;

namespace Tests.BLAS.CPU {
    public class ElementwiseSingleParamTests {
        void Run(Action<FloatCPUTensorBuffer, float, FloatCPUTensorBuffer> compute, float p, Func<float, float, float> singleCompute, float inputMin = -1, float inputMax = 1) {
            int[] shape = { 50 };
            FloatCPUTensorBuffer a = new FloatCPUTensorBuffer(shape);
            FloatCPUTensorBuffer e = new FloatCPUTensorBuffer(shape);
            FloatCPUTensorBuffer r = new FloatCPUTensorBuffer(shape);

            FloatTensor at = new FloatTensor(shape);
            FloatTensor et = new FloatTensor(shape);

            for (int i = 0; i < at.size; i++) {
                at.data[i] = UnityEngine.Random.Range(inputMin, inputMax);
                et.data[i] = singleCompute(at.data[i], p);
            }
            e.CopyFrom(et);
            a.CopyFrom(at);


            compute(a, p, r);
            CollectionAssert.AreEquivalent(e.buffer, r.buffer);
            a.Dispose();
            e.Dispose();
            r.Dispose();
        }

        [Test]
        public void Add() {
            Run((a, p, r) => DumbML.BLAS.CPU.ElementWiseFloatParam.Add(a, r, p),
                2,
                (a, p) => a + p);
        }
        [Test]
        public void Min() {
            Run((a, p, r) => DumbML.BLAS.CPU.ElementWiseFloatParam.Min(a, r, p),
                2,
                (a, p) => a < p ? a : p);
        }
        [Test]
        public void Max() {
            Run((a, p, r) => DumbML.BLAS.CPU.ElementWiseFloatParam.Max(a, r, p),
                2,
                (a, p) => a < p ? p : a);
        }
    }
}
