using UnityEngine;

namespace DumbML.BLAS.GPU {
    public static class Reduction {
        static void Compute(FloatGPUTensorBuffer input, int[] axis, FloatGPUTensorBuffer output, string kernel) {
            output.ExpandBuffer(input.size);

            Transpose(input, axis, output);


            ComputeShader shader = Kernels.reduction;

            int kernelID = shader.FindKernel(kernel);
            shader.SetBuffer(kernelID, Shader.PropertyToID("buffer"), output.buffer);
            shader.SetInt(Shader.PropertyToID("count"), input.size);
            shader.SetInt(Shader.PropertyToID("rstride"), output.size);
            shader.SetInt(Shader.PropertyToID("rcount"), input.size / output.size);
            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = output.size + (int)numThreads - 1;
          
            
            int s2 = input.size / output.size;

            for (int s = (s2 + 1) / 2; s > 0; s = (s + 1) / 2) {
                shader.SetInt(Shader.PropertyToID("s"), s);
                shader.SetInt(Shader.PropertyToID("s2"), s2);
                s2 = s;

                shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
                if (s == 1) {
                    break;
                }
            }

        }
        public static void Max(FloatGPUTensorBuffer input, int[] axis, FloatGPUTensorBuffer output) { 
            Compute(input, axis, output, "Max");
        }
        public static void Sum(FloatGPUTensorBuffer input, int[] axis, FloatGPUTensorBuffer output) { 
            Compute(input, axis, output, "Sum");
        }

        static void Transpose(FloatGPUTensorBuffer input, int[] raxis, FloatGPUTensorBuffer output) {
            var perm = Utils.GetIntArr();
            var strides = Utils.GetIntArr();
            GetPerm(perm);
            GetStrides(input.shape, strides);

            ComputeShader shader = Kernels.transpose;
            int kernelID = shader.FindKernel("Transpose");

            shader.SetBuffer(kernelID, "input", input.buffer);
            shader.SetBuffer(kernelID, "output", output.buffer);


            shader.SetInt("count", input.size);
            shader.SetInt("rank", input.Rank());
            shader.SetInts("ishape", input.shape);
            shader.SetInts("perm", perm);
            shader.SetInts("istrides", strides);


            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = input.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);

            Utils.Return(perm);
            Utils.Return(strides);

            void GetPerm(int[] result) {
                int a = 0;
                int b = raxis?.Length ?? -1;

                for (int i = 0; i < input.Rank(); i++) {
                    if (raxis == null || raxis.Contains(i)) {
                        result[a] = i;
                        a++;
                    }
                    else {
                        result[b] = i;
                        b++;
                    }
                }

            }

            void GetStrides(int[] shape, int[] result) {

                int stride = 1;
                for (int i = shape.Length - 1; i >= 0; i--) {

                    result[i] = stride;

                    int dimSize = shape[i];
                    stride *= dimSize;
                }
            }
        }

        static bool Contains(this int[] arr, int val) {
            foreach (var i in arr) {
                if (i == val) {
                    return true;
                }
            }
            return false;
        }
    }
}
