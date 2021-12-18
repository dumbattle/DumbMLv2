using UnityEngine;

namespace DumbML.BLAS.GPU {
    public static class Reduction {
        public static void Sum(FloatGPUTensorBuffer input, int[] axis, FloatGPUTensorBuffer output) {
            output.SetMinSize(input.size);
            Transpose(input, axis, output);

            ComputeShader shader = Kernels.reduction;


            int kernelID = shader.FindKernel("Sum");
            shader.SetBuffer(kernelID, Shader.PropertyToID("buffer"), output.buffer);

            shader.SetInt(Shader.PropertyToID("count"), input.size);
            shader.SetInt(Shader.PropertyToID("rstride"), output.size);
            shader.SetInt(Shader.PropertyToID("rcount"), input.size / output.size);


            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = input.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);

        }


        static void Transpose(FloatGPUTensorBuffer input, int[] raxis, FloatGPUTensorBuffer output) {
            // transpose input to [{reduced axis}, {non-reduced axis}]
            //     keep order of non-reduced axis the same
            //     this way no need to transpose back into correct shape


            ComputeShader shader = Kernels.transpose;
            int kernelID = shader.FindKernel("Transpose");

            shader.SetBuffer(kernelID, "input", input.buffer);
            shader.SetBuffer(kernelID, "output", output.buffer);


            shader.SetInt("count", input.size);
            shader.SetInt("rank", input.Rank());
            shader.SetInts("ishape", input.shape);
            shader.SetInts("perm", GetPerm());
            shader.SetInts("istrides", GetStrides(input.shape));


            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = input.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);

           
            int[] GetPerm() {
                int[] result = Utils.intArr;

                int a = 0;
                int b = raxis.Length;

                for (int i = 0; i < input.Rank(); i++) {
                    if (raxis.Contains(i)) {
                        result[a] = i;
                        a++;
                    }
                    else {
                        result[b] = i;
                        b++;
                    }
                }

                return result;
            }

            int[] GetStrides(int[] shape) {
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
