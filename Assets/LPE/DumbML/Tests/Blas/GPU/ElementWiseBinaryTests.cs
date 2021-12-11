using System;
using System.Collections.Generic;
using NUnit.Framework;
using UnityEngine;
using UnityEngine.TestTools;
using DumbML;

namespace Tests.BLAS {
    namespace GPU {
        public class ElementWiseBinaryTests {
            static void Run(Action<FloatGPUTensorBuffer, FloatGPUTensorBuffer, FloatGPUTensorBuffer> GPUCall, Action<FloatCPUTensorBuffer, FloatCPUTensorBuffer, FloatCPUTensorBuffer> CPPUOp, float eps = 1e-5f) {
                Run(new[] { 3, 4, 2 }, new[] { 3, 4, 2 }, new[] { 3, 4, 2 });
                Run(new[] { 5, 1, 5, 5 }, new[] { 5, 1, 5 }, new[] { 5, 5, 5, 5 });
                
                void Run(int[] shapeL, int[] shapeR, int[] shapeO) {
                    FloatGPUTensorBuffer leftGPU = new FloatGPUTensorBuffer(shapeL);
                    FloatGPUTensorBuffer rightGPU = new FloatGPUTensorBuffer(shapeR);
                    FloatGPUTensorBuffer outputGPU = new FloatGPUTensorBuffer(shapeO);

                    FloatCPUTensorBuffer leftCPU = new FloatCPUTensorBuffer(shapeL);
                    FloatCPUTensorBuffer rightCPU = new FloatCPUTensorBuffer(shapeR);
                    FloatCPUTensorBuffer outputCPU = new FloatCPUTensorBuffer(shapeO);
                    FloatCPUTensorBuffer outputGPU2CPU = new FloatCPUTensorBuffer(shapeO);

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
                    CPPUOp(leftCPU, rightCPU, outputCPU);
                    CollectionAssert.AreEqual(outputCPU.buffer, outputGPU2CPU.buffer);


                    leftGPU.Dispose();
                    rightGPU.Dispose();
                    outputGPU.Dispose();
                }
            }

            [Test]
            public static void Add() {
                Run((l, r, o) => DumbML.BLAS.GPU.ElementwiseBinary.Add(l, r, o),
                    (l, r, o) => {
                        DumbML.BLAS.CPU.ElementwiseBinary.Add(l, r, o);
                    }
                );
            }
            [Test]
            public static void Multiply() {
                Run((l, r, o) => DumbML.BLAS.GPU.ElementwiseBinary.Multiply(l, r, o),
                    (l, r, o) => {
                        DumbML.BLAS.CPU.ElementwiseBinary.Multiply(l, r, o);
                    }
                );
            }
            [Test]
            public static void Subtract() {
                Run((l, r, o) => DumbML.BLAS.GPU.ElementwiseBinary.Subtract(l, r, o),
                    (l, r, o) => {
                        DumbML.BLAS.CPU.ElementwiseBinary.Subtract(l, r, o);
                    }
                );
            }
        }
    }

}