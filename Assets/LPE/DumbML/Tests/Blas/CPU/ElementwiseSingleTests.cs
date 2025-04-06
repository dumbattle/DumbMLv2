using NUnit.Framework;
using DumbML;
using System;
using UnityEngine;

namespace Tests.BLAS.CPU {
    public class ElementwiseSingleTests {
        const float EPSILON = 1e-6f;

        void Run(Action<FloatCPUTensorBuffer, FloatCPUTensorBuffer> compute, Func<float, float> singleCompute, float inputMin = -1, float inputMax = 1) {
            int[] shape = { 5, 5, 5, 5 };

            FloatCPUTensorBuffer a = new FloatCPUTensorBuffer(shape);
            FloatCPUTensorBuffer e = new FloatCPUTensorBuffer(shape);

            FloatTensor at = new FloatTensor(shape);
            FloatTensor et = new FloatTensor(shape);

            for (int i = 0; i < at.size; i++) {
                at.data[i] = UnityEngine.Random.Range(inputMin, inputMax);
                et.data[i] = singleCompute(at.data[i]);
            }
            a.CopyFrom(at);
            e.CopyFrom(et);

            FloatCPUTensorBuffer r = new FloatCPUTensorBuffer(a.shape);
            compute(a, r);
            for (int i = 0; i < at.size; i++) {
                var dif = e.buffer[i] - r.buffer[i];

                if (Mathf.Abs(dif) > EPSILON) {
                    Assert.Fail($"Expected: {e.buffer.ContentString()} \nGot: {r.buffer.ContentString()}");
                    break;
                }
            }

            a.Dispose();
            e.Dispose();
            r.Dispose();
        }



        [Test]
        public void Copy() {
            Run((a, r) => DumbML.BLAS.CPU.ElementwiseSingle.Copy(a, r), (a) => a);
        }

        [Test]
        public void Exp() {
            Run((a, r) => DumbML.BLAS.CPU.ElementwiseSingle.Exp(a, r), (a) => Mathf.Exp(a));
        }

        [Test]
        public void Log() {
            Run((a, r) => DumbML.BLAS.CPU.ElementwiseSingle.Log(a, r), (a) => Mathf.Log(a), 0.1f, 1f);
        }
        [Test]
        public void ReLU() {
            Run((a, r) => DumbML.BLAS.CPU.ElementwiseSingle.ReLU(a, r), (a) => a < 0 ? 0 : a);
        }

        
        [Test]
        public void Sqr() {
            Run((a, r) => DumbML.BLAS.CPU.ElementwiseSingle.Sqr(a, r), (a) => a * a);
        }


        [Test]
        public void Sqrt() {
            Run((a, r) => DumbML.BLAS.CPU.ElementwiseSingle.Sqrt(a, r), (a) => Mathf.Sqrt(a), 0.1f, 1f);
        }
    }
}
