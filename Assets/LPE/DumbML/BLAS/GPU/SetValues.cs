using UnityEngine;

namespace DumbML.BLAS.GPU {
    public static class SetValues {
        public static void Zero(IntGPUTensorBuffer input) {
            ComputeShader shader = Kernels.setValues;
            ComputeBuffer inputBuffer = input.buffer;
            int kernelID = shader.FindKernel("Zeroi");
            shader.SetBuffer(kernelID, Shader.PropertyToID("input_i"), inputBuffer);
            shader.SetInt(Shader.PropertyToID("count"), input.size);

            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = input.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }
        public static void Zero(FloatGPUTensorBuffer input) {
            ComputeShader shader = Kernels.setValues;
            ComputeBuffer inputBuffer = input.buffer;
            int kernelID = shader.FindKernel("Zerof");
            shader.SetBuffer(kernelID, Shader.PropertyToID("input_f"), inputBuffer);
            shader.SetInt(Shader.PropertyToID("count"), input.size);

            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = input.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }
        public static void One(FloatGPUTensorBuffer input) {
            ComputeShader shader = Kernels.setValues;
            ComputeBuffer inputBuffer = input.buffer;
            int kernelID = shader.FindKernel("Onef");
            shader.SetBuffer(kernelID, Shader.PropertyToID("input_f"), inputBuffer);
            shader.SetInt(Shader.PropertyToID("count"), input.size);

            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = input.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }
    }


}

