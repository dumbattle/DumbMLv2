using NUnit.Framework;
using UnityEngine;
using DumbML;
using System;


namespace Tests {
    namespace GPU {
        public class ElementwiseSingleTests {
            static void Run(Action<GPUTensorBuffer, GPUTensorBuffer> GPUCall, Action<CPUTensorBuffer, CPUTensorBuffer> CPPUOp, float eps = 1e-5f) {
                GPUTensorBuffer inputGPU = new GPUTensorBuffer(3, 4, 2);
                GPUTensorBuffer outputGPU = new GPUTensorBuffer(3, 4, 2);

                CPUTensorBuffer inputCPU = new CPUTensorBuffer(3, 4, 2);
                CPUTensorBuffer outputCPU = new CPUTensorBuffer(3, 4, 2);
                for (int i = 0; i < inputCPU.size; i++) {
                    inputCPU.buffer[i] = UnityEngine.Random.Range(-1f, 1f);
                }
                CPUTensorBuffer outputGPU2CPU = new CPUTensorBuffer(3, 4, 2);
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
