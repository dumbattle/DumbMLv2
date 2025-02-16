﻿using UnityEngine;

namespace DumbML.BLAS.GPU {
    public static class Reduction {
        public static void Sum(FloatGPUTensorBuffer input, int[] axis, FloatGPUTensorBuffer output) {
            output.ExpandBuffer(input.size);

            //if (axis != null) {
                Transpose(input, axis, output);
            //}
            //else {
            //    // this doesn't work????
            //    // ElementwiseSingle uses length of output
            //    ElementwiseSingle.Copy(input, output, true);
            //}

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
