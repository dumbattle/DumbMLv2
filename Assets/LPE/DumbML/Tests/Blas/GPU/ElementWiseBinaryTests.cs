using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using DumbML;

namespace Tests.BLAS {
    namespace GPU {
        public class ElementWiseBinaryTests {
            static void Run(Action<FloatGPUTensorBuffer, FloatGPUTensorBuffer, FloatGPUTensorBuffer> GPUCall, Func<FloatCPUTensorBuffer, FloatCPUTensorBuffer, FloatCPUTensorBuffer> CPPUOp, float eps = 1e-5f) {
                FloatGPUTensorBuffer leftGPU = new FloatGPUTensorBuffer(3, 4, 2);
                FloatGPUTensorBuffer rightGPU = new FloatGPUTensorBuffer(3, 4, 2);
                FloatGPUTensorBuffer outputGPU = new FloatGPUTensorBuffer(3, 4, 2);

                FloatCPUTensorBuffer leftCPU = new FloatCPUTensorBuffer(3, 4, 2);
                FloatCPUTensorBuffer rightCPU = new FloatCPUTensorBuffer(3, 4, 2);
                FloatCPUTensorBuffer outputGPU2CPU = new FloatCPUTensorBuffer(3, 4, 2);

                for (int i = 0; i < leftCPU.size; i++) {
                    leftCPU.buffer[i] = UnityEngine.Random.Range(-1f, 1f);
                }
                for (int i = 0; i < rightCPU.size; i++) {
                    rightCPU.buffer[i] = UnityEngine.Random.Range(-1f, 1f);
                }


                leftGPU.CopyFrom(leftCPU);
                rightGPU.CopyFrom(rightCPU);

                GPUCall(leftGPU, rightGPU, outputGPU);
                outputGPU.CopyTo(outputGPU2CPU);
                var outputCPU = CPPUOp(leftCPU, rightCPU);
                CollectionAssert.AreEqual(outputCPU.buffer, outputGPU2CPU.buffer);


                leftGPU.Dispose();
                rightGPU.Dispose();
                outputGPU.Dispose();
            }

            [Test]
            public static void Add() {
                Run((l, r, o) => DumbML.BLAS.GPU.ElementwiseBinary.Add(l, r, o),
                    (l, r) => {
                        FloatCPUTensorBuffer result = new FloatCPUTensorBuffer(l.shape);
                        DumbML.BLAS.CPU.ElementwiseBinary.Add(l, r, result);
                        return result;
                    }
                );
            }
        }
    }

}