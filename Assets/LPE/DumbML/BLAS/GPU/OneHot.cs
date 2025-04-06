using UnityEngine;


namespace DumbML.BLAS.GPU {
    public static class OneHot {
        public static void Compute(IntGPUTensorBuffer input, int depth, float on, float off, FloatGPUTensorBuffer output) {
            // TODO - check shapes
            ComputeShader shader = Kernels.oneHot;
            int kernelID = shader.FindKernel("OneHot");

            shader.SetBuffer(kernelID, "inds", input.buffer);
            shader.SetBuffer(kernelID, "output", output.buffer);
            shader.SetInt("count", output.size);
            shader.SetInt("depth", depth);
            shader.SetFloat("onval", on);
            shader.SetFloat("offval", off);

            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = output.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }
    }
}