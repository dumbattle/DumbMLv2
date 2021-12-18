using UnityEngine;

namespace DumbML.BLAS.GPU {
    public static class Transpose {
        public static void Compute(FloatGPUTensorBuffer input, int[] perm, FloatGPUTensorBuffer output) {
            ComputeShader shader = Kernels.transpose;
            int kernelID = shader.FindKernel("Transpose");

            shader.SetBuffer(kernelID, "input", input.buffer);
            shader.SetBuffer(kernelID, "output", output.buffer);

            shader.SetInt("count", input.size);
            shader.SetInt("rank", input.Rank());
            shader.SetInts("ishape", input.shape);
            shader.SetInts("perm", perm);
            shader.SetInts("istrides", GetStrides(input.shape));

            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = input.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }

        static int[] GetStrides(int[] shape) {
            int[] result = Utils.intArr;

            int stride = 1;
            for (int i = shape.Length - 1; i >= 0; i--) {

                result[i] = stride;

                int dimSize = shape[i];
                stride *= dimSize;
            }
            return result;
        }
    }

    public static class Utils {
        public static int[] intArr { get; private set; } = new int[64];
    }
}