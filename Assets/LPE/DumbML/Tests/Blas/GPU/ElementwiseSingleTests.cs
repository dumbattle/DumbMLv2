using NUnit.Framework;
using UnityEngine;
using DumbML;
using System;


namespace Tests {
    namespace GPU {
        public class ElementwiseSingleTests {
            static void Run(Action<FloatGPUTensorBuffer, FloatGPUTensorBuffer> GPUCall, Action<FloatCPUTensorBuffer, FloatCPUTensorBuffer> CPPUOp, float eps = 1e-5f) {
                FloatGPUTensorBuffer inputGPU = new FloatGPUTensorBuffer(3, 4, 2);
                FloatGPUTensorBuffer outputGPU = new FloatGPUTensorBuffer(3, 4, 2);

                FloatCPUTensorBuffer inputCPU = new FloatCPUTensorBuffer(3, 4, 2);
                FloatCPUTensorBuffer outputCPU = new FloatCPUTensorBuffer(3, 4, 2);
                for (int i = 0; i < inputCPU.size; i++) {
                    inputCPU.buffer[i] = UnityEngine.Random.Range(-1f, 1f);
                }
                FloatCPUTensorBuffer outputGPU2CPU = new FloatCPUTensorBuffer(3, 4, 2);
                inputGPU.CopyFrom(inputCPU);


                GPUCall(inputGPU, outputGPU);
                outputGPU.CopyTo(outputGPU2CPU);
                CPPUOp(inputCPU, outputCPU);
                CollectionAssert.AreEqual(outputCPU.buffer, outputGPU2CPU.buffer);
                inputGPU.Dispose();
                outputGPU.Dispose();
            }

            //[Test]
            //public void Abs() {
            //    Run((a, b) => DumbML.BLAS.GPU.ElementwiseSingle.Abs(a, b),
            //        (x) => new Abs(x));
            //}

            //[Test]
            //public void Exp() {
            //    Run((a, b) => DumbML.BLAS.GPU.ElementwiseSingle.Exp(a, b),
            //        (x) => new Exp(x));
            //}

            //[Test]
            //public void Log() {
            //    Run((a, b) => DumbML.BLAS.GPU.ElementwiseSingle.Log(a, b),
            //        (x) => new Log(x));
            //}

            //[Test]
            //public void ReLU() {
            //    Run((a, b) => DumbML.BLAS.GPU.ElementwiseSingle.ReLU(a, b),
            //        (x) => new Relu(x));
            //}
        }
    }

}
