using UnityEngine;

namespace DumbML.BLAS.GPU {
    public static class ElementwiseSingleParam {
        static void Call(FloatGPUTensorBuffer input, FloatGPUTensorBuffer output, string kernelName, string inplaceName, float p) {
            if (output == input) {
                Call_Inplace(input, inplaceName, p);
            }
            else {
                Call_Normal(input, output, kernelName, p);
            }
        }

        static void Call_Inplace(FloatGPUTensorBuffer input, string kernelName, float p) {
            ComputeShader shader = Kernels.elementWiseSingleParam;
            ComputeBuffer inputBuffer = input.buffer;
            int kernelID = shader.FindKernel(kernelName);
            shader.SetBuffer(kernelID, Shader.PropertyToID("input"), inputBuffer);
            shader.SetFloat(Shader.PropertyToID("p"), p);
            shader.SetInt(Shader.PropertyToID("count"), input.size);
            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = input.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }
        static void Call_Normal(FloatGPUTensorBuffer input, FloatGPUTensorBuffer output, string kernelName, float p) {
            if (!input.shape.CompareContents(output.shape)) {
                throw new System.ArgumentException($"Input and output tensors do not have same shape: {input.shape.ContentString()} vs {output.shape.ContentString()}");
            }
            ComputeShader shader = Kernels.elementWiseSingleParam;
            ComputeBuffer inputBuffer = input.buffer;
            ComputeBuffer outputBuffer = output.buffer;
            int kernelID = shader.FindKernel(kernelName);
            shader.SetBuffer(kernelID, Shader.PropertyToID("input"), inputBuffer);
            shader.SetBuffer(kernelID, Shader.PropertyToID("output"), outputBuffer);
            shader.SetFloat(Shader.PropertyToID("p"), p);
            shader.SetInt(Shader.PropertyToID("count"), output.size);
            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = input.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }



        public static void Add(FloatGPUTensorBuffer input, FloatGPUTensorBuffer output, float v) {
            const string name = "Add";
            const string inplace = name + "_Inplace";

            Call(input, output, name, inplace, v);
        }

        public static void Multiply(FloatGPUTensorBuffer input, FloatGPUTensorBuffer output, float v) {
            const string name = "Multiply";
            const string inplace = name + "_Inplace";

            Call(input, output, name, inplace, v);
        }
        
        public static void Max(FloatGPUTensorBuffer input, FloatGPUTensorBuffer output, float v) {
            const string name = "Max";
            const string inplace = name + "_Inplace";

            Call(input, output, name, inplace, v);
        }
        public static void Min(FloatGPUTensorBuffer input, FloatGPUTensorBuffer output, float v) {
            const string name = "Min";
            const string inplace = name + "_Inplace";

            Call(input, output, name, inplace, v);
        }
    }
}
