using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using DumbML;

namespace Tests.BLAS {
    namespace GPU {
        public class ElementWiseBinaryTests {
            static void Run(Action<GPUTensorBuffer, GPUTensorBuffer, GPUTensorBuffer> GPUCall, Func<CPUTensorBuffer, CPUTensorBuffer, CPUTensorBuffer> CPPUOp, float eps = 1e-5f) {
                GPUTensorBuffer leftGPU = new GPUTensorBuffer(3, 4, 2);
                GPUTensorBuffer rightGPU = new GPUTensorBuffer(3, 4, 2);
                GPUTensorBuffer outputGPU = new GPUTensorBuffer(3, 4, 2);

                CPUTensorBuffer leftCPU = new CPUTensorBuffer(3, 4, 2);
                CPUTensorBuffer rightCPU = new CPUTensorBuffer(3, 4, 2);
                CPUTensorBuffer outputGPU2CPU = new CPUTensorBuffer(3, 4, 2);

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
                        CPUTensorBuffer result = new CPUTensorBuffer(l.shape);
                        DumbML.BLAS.CPU.ElementwiseBinary.Add(l, r, result);
                        return result;
                    }
                );
            }
        }
    }

}