using NUnit.Framework;
using UnityEngine;
using DumbML;
using System;


namespace Tests.BLAS {
    namespace GPU {

        public class SamplingTests {
            [Test]
            public void SampleCategorical() {
                FloatGPUTensorBuffer inputGPU = new FloatGPUTensorBuffer(1, 3);
                IntGPUTensorBuffer outputGPU = new IntGPUTensorBuffer(1);

                FloatCPUTensorBuffer inputCPU = new FloatCPUTensorBuffer(1, 3);
                IntCPUTensorBuffer outputGPU2CPU = new IntCPUTensorBuffer(1);


                for (int i = 0; i < inputCPU.size; i++) {
                    inputCPU.buffer[i] = 0.333333f;
                }
                inputGPU.CopyFrom(inputCPU);

                int[] result = new int[3];

                for (int i = 0; i < 10000; i++) {
                    DumbML.BLAS.GPU.SampleCategorical.Compute(inputGPU, outputGPU);
                    outputGPU.CopyTo(outputGPU2CPU);
                    result[outputGPU2CPU.buffer[0]]++;
                }
                Debug.Log(result.ContentString());

                inputGPU.Dispose();
                outputGPU.Dispose();
                inputCPU.Dispose();
                outputGPU2CPU.Dispose();
            }
        }


        public class ElementwiseSingleTests {
            static void Run(Action<FloatGPUTensorBuffer, FloatGPUTensorBuffer> GPUCall, Action<FloatCPUTensorBuffer, FloatCPUTensorBuffer> CPUCall, float min = -1, float max = 1, float eps = 1e-5f) {
                RunNormal(GPUCall, CPUCall, min, max, eps);

                static void RunNormal(Action<FloatGPUTensorBuffer, FloatGPUTensorBuffer> GPUCall, Action<FloatCPUTensorBuffer, FloatCPUTensorBuffer> CPUCall, float min, float max, float eps) {
                    int[] shape = { 5, 5, 5 };
                    FloatGPUTensorBuffer inputGPU = new FloatGPUTensorBuffer(shape);
                    FloatGPUTensorBuffer outputGPU = new FloatGPUTensorBuffer(shape);

                    FloatCPUTensorBuffer inputCPU = new FloatCPUTensorBuffer(shape);
                    FloatCPUTensorBuffer outputCPU = new FloatCPUTensorBuffer(shape);

                    for (int i = 0; i < inputCPU.size; i++) {
                        inputCPU.buffer[i] = UnityEngine.Random.Range(min, max);
                    }

                    FloatCPUTensorBuffer outputGPU2CPU = new FloatCPUTensorBuffer(shape);
                    FloatCPUTensorBuffer inputGPU2CPU = new FloatCPUTensorBuffer(shape);
                    inputGPU.CopyFrom(inputCPU);


                    GPUCall(inputGPU, outputGPU);
                    GPUCall(inputGPU, inputGPU);
                    CPUCall(inputCPU, outputCPU);
                    outputGPU.CopyTo(outputGPU2CPU);
                    inputGPU.CopyTo(inputGPU2CPU);
                    for (int i = 0; i < outputCPU.size; i++) {
                        var trueVal = outputCPU.buffer[i];
                        var output1 = outputGPU2CPU.buffer[i];
                        var outputInplace = inputGPU2CPU.buffer[i];
                        var dif1 = trueVal - output1;
                        var dif2 = trueVal - outputInplace;
                        Assert.IsTrue(Mathf.Abs(dif1) < eps, $"{i}: {trueVal} - {output1}");
                        Assert.IsTrue(Mathf.Abs(dif2) < eps, $"{i}: {trueVal} - {outputInplace}");
                    }
                    inputGPU.Dispose();
                    outputGPU.Dispose();
                    outputGPU2CPU.Dispose();
                    inputGPU2CPU.Dispose();
                    inputCPU.Dispose();
                    outputCPU.Dispose();
                }
             
            }

            //[Test]
            //public void Abs() {
            //    Run((a, b) => DumbML.BLAS.GPU.ElementwiseSingle.Abs(a, b),
            //        (a, b) => DumbML.BLAS.CPU.ElementwiseSingle.Abs(a, b));
            //}

            [Test]
            public void Exp() {
                Run((a, b) => DumbML.BLAS.GPU.ElementwiseSingle.Exp(a, b),
                    (a, b) => DumbML.BLAS.CPU.ElementwiseSingle.Exp(a, b), eps: 1e-1f);
            }

            [Test]
            public void Log() {
                Run((a, b) => DumbML.BLAS.GPU.ElementwiseSingle.Log(a, b),
                    (a, b) => DumbML.BLAS.CPU.ElementwiseSingle.Log(a, b), .1f, 2f);
            }

            [Test]
            public void ReLU() {
                Run((a, b) => DumbML.BLAS.GPU.ElementwiseSingle.ReLU(a, b),
                    (a, b) => DumbML.BLAS.CPU.ElementwiseSingle.ReLU(a, b));
            }
            [Test]
            public void Sqr() {
                Run((a, b) => DumbML.BLAS.GPU.ElementwiseSingle.Sqr(a, b),
                    (a, b) => DumbML.BLAS.CPU.ElementwiseSingle.Sqr(a, b));
            }
            [Test]
            public void Sqrt() {
                Run((a, b) => DumbML.BLAS.GPU.ElementwiseSingle.Sqrt(a, b),
                    (a, b) => DumbML.BLAS.CPU.ElementwiseSingle.Sqrt(a, b), .1f, 2f);
            }
        }
    }
}
