using UnityEngine;

namespace DumbML.BLAS.GPU {
    public static class ElementwiseSingle {
        private const string INPLACE_SUFFIX = "_Inplace";

        struct Names {
            public string normal;
            public string inplace;
        }

        static void Call(FloatGPUTensorBuffer input, FloatGPUTensorBuffer output, Names names, bool ignoreShape = false) {
            if (!ignoreShape && !input.shape.CompareContents(output.shape)) {
                throw new System.ArgumentException($"Input and output tensors do not have same shape: {input.shape.ContentString()} vs {output.shape.ContentString()}");
            }
           
            if (input.Equals(output)) { 
                Call_Inplace(input, names.inplace, ignoreShape);
            }
            else {
                Call_Normal(input, output, names.normal, ignoreShape);
            }
        }

        static void Call_Normal(FloatGPUTensorBuffer input, FloatGPUTensorBuffer output, string kernelName, bool ignoreShape = false) {
            ComputeShader shader = Kernels.elementWiseSingle;
            ComputeBuffer inputBuffer = input.buffer;
            ComputeBuffer outputBuffer = output.buffer;
            int kernelID = shader.FindKernel(kernelName);
            shader.SetBuffer(kernelID, Shader.PropertyToID("input"), inputBuffer);
            shader.SetBuffer(kernelID, Shader.PropertyToID("output"), outputBuffer);
            shader.SetInt(Shader.PropertyToID("count"), input.size);

            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = input.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }
        static void Call_Inplace(FloatGPUTensorBuffer buffer, string kernelName, bool ignoreShape = false) {
            ComputeShader shader = Kernels.elementWiseSingle;
            ComputeBuffer inputBuffer = buffer.buffer;
            int kernelID = shader.FindKernel(kernelName);
            shader.SetBuffer(kernelID, Shader.PropertyToID("input"), inputBuffer);
            shader.SetInt(Shader.PropertyToID("count"), buffer.size);

            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = buffer.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }


        public static void Abs(FloatGPUTensorBuffer input, FloatGPUTensorBuffer output) {
            const string NAME = "Abs";
            const string INPLACE = NAME + INPLACE_SUFFIX;
            Names n;
            n.normal = NAME;
            n.inplace = INPLACE;

            Call(input, output, n);
        }
        public static void Copy(FloatGPUTensorBuffer input, FloatGPUTensorBuffer output, bool ignoreShape = false) {
            const string NAME = "Copy";
            const string INPLACE = NAME + INPLACE_SUFFIX;
            Names n;
            n.normal = NAME;
            n.inplace = INPLACE;

            Call(input, output, n, ignoreShape);
            
        }
        public static void Exp(FloatGPUTensorBuffer input, FloatGPUTensorBuffer output) {
            const string NAME = "Exp";
            const string INPLACE = NAME + INPLACE_SUFFIX;
            Names n;
            n.normal = NAME;
            n.inplace = INPLACE;

            Call(input, output, n);
        }
        public static void Log(FloatGPUTensorBuffer input, FloatGPUTensorBuffer output) {
            const string NAME = "Log";
            const string INPLACE = NAME + INPLACE_SUFFIX;
            Names n;
            n.normal = NAME;
            n.inplace = INPLACE;

            Call(input, output, n);
        }
        public static void ReLU(FloatGPUTensorBuffer input, FloatGPUTensorBuffer output) {
            const string NAME = "ReLU";
            const string INPLACE = NAME + INPLACE_SUFFIX;
            Names n;
            n.normal = NAME;
            n.inplace = INPLACE;

            Call(input, output, n);
        }
        public static void Sqr(FloatGPUTensorBuffer input, FloatGPUTensorBuffer output) {
            const string NAME = "Sqr";
            const string INPLACE = NAME + INPLACE_SUFFIX;
            Names n;
            n.normal = NAME;
            n.inplace = INPLACE;

            Call(input, output, n);
        } 
        public static void Sqrt(FloatGPUTensorBuffer input, FloatGPUTensorBuffer output) {
            const string NAME = "Sqrt";
            const string INPLACE = NAME + INPLACE_SUFFIX;
            Names n;
            n.normal = NAME;
            n.inplace = INPLACE;

            Call(input, output, n);
        }
    }
}
