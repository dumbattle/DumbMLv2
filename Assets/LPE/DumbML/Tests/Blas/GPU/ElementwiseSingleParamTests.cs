using System;
using NUnit.Framework;
using DumbML;
using UnityEngine;


namespace Tests.BLAS {
    namespace GPU {
        public class ElementwiseSingleParamTests {
            static void Run(Action<GPUTensorBuffer, GPUTensorBuffer, float> GPUCall, Func<CPUTensorBuffer, float, CPUTensorBuffer> CPPUOp, float param, float eps = 1e-5f) {
                GPUTensorBuffer inputGPU = new GPUTensorBuffer(3, 4, 2);
                GPUTensorBuffer outputGPU = new GPUTensorBuffer(3, 4, 2);

                CPUTensorBuffer inputCPU = new CPUTensorBuffer(3, 4, 2);
                for (int i = 0; i < inputCPU.size; i++) {
                    inputCPU.buffer[i] = UnityEngine.Random.Range(-1f, 1f);
                }
                CPUTensorBuffer outputGPU2CPU = new CPUTensorBuffer(3, 4, 2);

                inputGPU.CopyFrom(inputCPU);

                GPUCall(inputGPU, outputGPU, param);
                outputGPU.CopyTo(outputGPU2CPU);
                var outputCPU = CPPUOp(inputCPU, param);
                CollectionAssert.AreEqual(outputCPU.buffer, outputGPU2CPU.buffer);
                inputGPU.Dispose();
                outputGPU.Dispose();
            }

            [Test]
            public void Add() {
                float v = 4;
                Run((a, b, p) => DumbML.BLAS.GPU.ElementwiseSingleParam.Add(a, b, p),
                    (x, p) => {
                        CPUTensorBuffer result = new CPUTensorBuffer(x.shape);
                        DumbML.BLAS.CPU.ElementWiseFloatParam.Add(x, result, p);
                        return result;
                    },
                    v);
            }
        }
    }

}