using UnityEngine;

namespace DumbML.BLAS.GPU {
    public static class Cast {
        const string BOOL_TO_FLOAT = "bool_float";
        const string FLOAT_TO_BOOL = "float_bool";
        const string FLOAT_TO_INT = "float_int";
        const string INT_TO_FLOAT = "int_float";
        const string INT_TO_BOOL = "int_bool";
        const string BOOL_TO_INT = "bool_int";

        const string BOOL_BUFFER = "buffer_bool";
        const string FLOAT_BUFFER = "buffer_float";
        const string INT_BUFFER = "buffer_int";


        static void Call<T, U>(GPUTensorBuffer<T> input, GPUTensorBuffer<U> output, string kernelName, string lbuffer, string rbuffer) where T : struct where U : struct {
            if (!input.shape.CompareContents(output.shape)) {
                throw new System.ArgumentException($"Input and output tensors do not have same shape: {input.shape.ContentString()} vs {output.shape.ContentString()}");
            }

            ComputeShader shader = Kernels.cast;
            ComputeBuffer inputBuffer = input.buffer;
            ComputeBuffer outputBuffer = output.buffer;

            int kernelID = shader.FindKernel(kernelName);

            shader.SetBuffer(kernelID, Shader.PropertyToID(lbuffer), inputBuffer);
            shader.SetBuffer(kernelID, Shader.PropertyToID(rbuffer), outputBuffer);
            shader.SetInt(Shader.PropertyToID("count"), input.size);

            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);

            int size = input.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }


        public static void BoolToFloat(GPUTensorBuffer<bool> src, GPUTensorBuffer<float> dest) {
            Call(src, dest, BOOL_TO_FLOAT, BOOL_BUFFER, FLOAT_BUFFER);
        }
        public static void FloatToBool(GPUTensorBuffer<float> src, GPUTensorBuffer<bool> dest) {
            Call(src, dest, FLOAT_TO_BOOL, FLOAT_BUFFER, BOOL_BUFFER);
        }


        public static void FloatToInt(GPUTensorBuffer<float> src, GPUTensorBuffer<int> dest) {
            Call(src, dest, FLOAT_TO_INT, FLOAT_BUFFER, INT_BUFFER);
        }
        public static void IntToFloat(GPUTensorBuffer<int> src, GPUTensorBuffer<float> dest) {
            Call(src, dest, INT_TO_FLOAT, INT_BUFFER, FLOAT_BUFFER);
        }


        public static void BoolToInt(GPUTensorBuffer<bool> src, GPUTensorBuffer<int> dest) {
            Call(src, dest, BOOL_TO_INT, BOOL_BUFFER, INT_BUFFER);
        }
        public static void IntToBool(GPUTensorBuffer<int> src, GPUTensorBuffer<bool> dest) {
            Call(src, dest, INT_TO_BOOL, INT_BUFFER, BOOL_BUFFER);
        }
    }
}
