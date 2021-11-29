using UnityEngine;

namespace DumbML.BLAS.GPU {
    public static class SetValues {
        public static void Zero(GPUTensorBuffer input) {
            ComputeShader shader = Kernels.setValues;
            ComputeBuffer inputBuffer = input.buffer;
            int kernelID = shader.FindKernel("Zero");
            shader.SetBuffer(kernelID, Shader.PropertyToID("input"), inputBuffer);
            shader.SetInt(Shader.PropertyToID("count"), input.size);

            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = input.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }
        public static void One(GPUTensorBuffer input) {
            ComputeShader shader = Kernels.setValues;
            ComputeBuffer inputBuffer = input.buffer;
            int kernelID = shader.FindKernel("One");
            shader.SetBuffer(kernelID, Shader.PropertyToID("input"), inputBuffer);
            shader.SetInt(Shader.PropertyToID("count"), input.size);

            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = input.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }
    }


}

