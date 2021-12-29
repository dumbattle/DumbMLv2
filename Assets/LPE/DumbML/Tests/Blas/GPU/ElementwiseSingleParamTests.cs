using System;
using NUnit.Framework;
using DumbML;
using UnityEngine;


namespace Tests.BLAS {
    namespace GPU {
        public class ElementwiseSingleParamTests {
            static void Run(Action<FloatGPUTensorBuffer, FloatGPUTensorBuffer, float> GPUCall, Func<FloatCPUTensorBuffer, float, FloatCPUTensorBuffer> CPPUOp, float param, float eps = 1e-5f) {
                FloatGPUTensorBuffer inputGPU = new FloatGPUTensorBuffer(3, 4, 2);
                FloatGPUTensorBuffer outputGPU = new FloatGPUTensorBuffer(3, 4, 2);

                FloatCPUTensorBuffer inputCPU = new FloatCPUTensorBuffer(3, 4, 2);
                for (int i = 0; i < inputCPU.size; i++) {
                    inputCPU.buffer[i] = UnityEngine.Random.Range(-1f, 1f);
                }
                FloatCPUTensorBuffer outputGPU2CPU = new FloatCPUTensorBuffer(3, 4, 2);

                inputGPU.CopyFrom(inputCPU);

                GPUCall(inputGPU, outputGPU, param);
                outputGPU.CopyTo(outputGPU2CPU);
                FloatCPUTensorBuffer outputCPU = CPPUOp(inputCPU, param);
                CollectionAssert.AreEquivalent(outputCPU.buffer, outputGPU2CPU.buffer);

                inputGPU.Dispose();
                outputGPU.Dispose();

                inputCPU.Dispose();
                outputGPU2CPU.Dispose();
                outputCPU.Dispose();
            }

            [Test]
            public void Add() {
                float v = 4;
                Run((a, b, p) => DumbML.BLAS.GPU.ElementwiseSingleParam.Add(a, b, p),
                    (x, p) => {
                        FloatCPUTensorBuffer result = new FloatCPUTensorBuffer(x.shape);
                        DumbML.BLAS.CPU.ElementWiseFloatParam.Add(x, result, p);
                        return result;
                    },
                    v);
            }
        }
    }

}