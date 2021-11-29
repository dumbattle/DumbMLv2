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
            CPUTensorBuffer a = new CPUTensorBuffer(5);
            a.CopyFrom(Tensor.FromArray(new float[] { 1, 2, 3, 4, 5 }));
            float b = 2;

            CPUTensorBuffer e = new CPUTensorBuffer(5);
            e.CopyFrom(Tensor.FromArray(new float[] { 3, 4, 5, 6, 7 }));

            CPUTensorBuffer r = new CPUTensorBuffer(a.shape);

            DumbML.BLAS.CPU.ElementWiseFloatParam.Add(a, r, b);
            CollectionAssert.AreEqual(e.buffer, r.buffer);
        }
    }
}
