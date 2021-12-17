using UnityEngine;

namespace DumbML.BLAS.GPU {
    public static class Reduction {
        public static void Sum(FloatGPUTensorBuffer input, int[] axis, FloatGPUTensorBuffer output) {
            // zero out output
            ElementwiseSingle.Copy(input, output, true);
            output.SetMinSize(input.size);

            ComputeShader shader = Kernels.reduction;
            ComputeBuffer inputBuffer = input.buffer;
            ComputeBuffer outputBuffer = output.buffer;

            int kernelID = shader.FindKernel("Sum");

            shader.SetBuffer(kernelID, Shader.PropertyToID("input"), inputBuffer);
            shader.SetBuffer(kernelID, Shader.PropertyToID("output"), outputBuffer);
            shader.SetInt(Shader.PropertyToID("count"), output.size);


            shader.SetInts("ishape", input.shape);
            shader.SetInts("raxis", axis);

            shader.SetInt("irank", input.Rank());
            shader.SetInt("rrank", axis.Length);
            shader.SetInt("isize", input.size);
            //shader.SetInt("osize", output.size);

            // temp for testing
            int osize = 1;
            for (int i = 0; i < input.shape.Length; i++) {
                if (System.Array.Exists(axis, (a) => a == i)) {
                    continue;
                }
                osize *= input.shape[i];
            }

            shader.SetInt("osize", osize);




            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = input.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);

        }

    }
}
