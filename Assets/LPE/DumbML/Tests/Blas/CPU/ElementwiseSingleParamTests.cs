using System.Collections;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using DumbML;

namespace Tests.BLAS.CPU {
    public class ElementwiseSingleParamTests {
        [Test]
        public void Add() {
            FloatCPUTensorBuffer a = new FloatCPUTensorBuffer(5);
            a.CopyFrom(FloatTensor.FromArray(new float[] { 1, 2, 3, 4, 5 }));
            float b = 2;

            FloatCPUTensorBuffer e = new FloatCPUTensorBuffer(5);
            e.CopyFrom(FloatTensor.FromArray(new float[] { 3, 4, 5, 6, 7 }));

            FloatCPUTensorBuffer r = new FloatCPUTensorBuffer(a.shape);

            DumbML.BLAS.CPU.ElementWiseFloatParam.Add(a, r, b);
            CollectionAssert.AreEquivalent(e.buffer, r.buffer);
            a.Dispose();
            e.Dispose();
            r.Dispose();
        }
    }
}
