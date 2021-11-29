﻿using UnityEngine;

namespace DumbML.BLAS.GPU {
    public static class ElementwiseSingle {
        static void Call(GPUTensorBuffer input, GPUTensorBuffer output, string kernelName) {
            if (!input.shape.CompareContents(output.shape)) {
                throw new System.ArgumentException($"Input and output tensors do not have same shape: {input.shape.ContentString()} vs {output.shape.ContentString()}");
            }
            ComputeShader shader = Kernels.elementWiseSingle;
            ComputeBuffer inputBuffer = input.buffer;
            ComputeBuffer outputBuffer = output.buffer;
            int kernelID = shader.FindKernel(kernelName);
            shader.SetBuffer(kernelID, Shader.PropertyToID("input"), inputBuffer);
            shader.SetBuffer(kernelID, Shader.PropertyToID("output"), outputBuffer);
            shader.SetInt(Shader.PropertyToID("count"), output.size);

            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = input.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }
       
        
        
        
        public static void Abs(GPUTensorBuffer input, GPUTensorBuffer output) {
            Call(input, output, "Abs");
        }
        public static void Copy(GPUTensorBuffer input, GPUTensorBuffer output) {
            Call(input, output, "Copy");
        }
        public static void Exp(GPUTensorBuffer input, GPUTensorBuffer output) {
            Call(input, output, "Exp");
        }
        public static void Log(GPUTensorBuffer input, GPUTensorBuffer output) {
            Call(input, output, "Log");
        }
        public static void ReLU(GPUTensorBuffer input, GPUTensorBuffer output) {
            Call(input, output, "ReLU");
        }
        public static void Sqr(GPUTensorBuffer input, GPUTensorBuffer output) {
            Call(input, output, "Sqr");
        }
    }


}

