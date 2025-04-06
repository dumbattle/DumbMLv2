using UnityEngine;

namespace DumbML.BLAS.GPU {
    public static class SampleCategorical {
        public static void Compute(FloatGPUTensorBuffer input, IntGPUTensorBuffer dest) {
            // Validate shape
            if (dest.Rank() != input.Rank() - 1) {
                throw new System.ArgumentException($"Incompatible Destination shapes for SampleCategorical\n\n");
            }

            for (int i = 0; i < dest.Rank(); i++) {
                if (dest.shape[i] != input.shape[i]) {
                    throw new System.ArgumentException($"Incompatible Destination shapes for SampleCategorical\n\n");
                }
            }

            ComputeShader shader = Kernels.sampleCategorical;
            ComputeBuffer inputBuffer = input.buffer;
            ComputeBuffer outputBuffer = dest.buffer;

            int kernelID = shader.FindKernel("SampleCategorical");

            shader.SetBuffer(kernelID, Shader.PropertyToID("input"), inputBuffer);
            shader.SetBuffer(kernelID, Shader.PropertyToID("output"), outputBuffer);
            shader.SetInt(Shader.PropertyToID("count"), dest.size);
            shader.SetInt(Shader.PropertyToID("stride"), input.shape[input.Rank() - 1]);
            shader.SetFloat(Shader.PropertyToID("seed"), Random.value);


            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = dest.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }
    }
}
