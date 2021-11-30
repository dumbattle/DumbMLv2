using UnityEngine;
using System;


namespace DumbML.BLAS.GPU {


    public static class ElementwiseBinary {
        private const string _InplaceL = "_InplaceL";
        private const string _InplaceR = "_InplaceR";
        private const string _Self = "_Self";
        private const string _SelfInplace = "_SelfInplace";

        struct Names {
            public string normal;
            public string inplaceL;
            public string inplaceR;
            public string self;
            public string selfInplace;
        }

        static void Call(GPUTensorBuffer left, GPUTensorBuffer right, GPUTensorBuffer output, Names names) {
            if (left == right) {
                if (output == left) {
                    Call_SelfInplace(left, names.selfInplace);
                }
                else {
                    Call_Self(left, output, names.self);
                }
            }
            else if (output == left) {
                Call_Inplace(left, right, names.inplaceL);
            }
            else if (output == right) {
                Call_Inplace(left, right, names.inplaceR);
            }
            else {
                Call_Normal(left, right, output, names.normal);
            }

        }
       
        
        static void Call_Normal(GPUTensorBuffer left, GPUTensorBuffer right, GPUTensorBuffer output, string kernelName) {
            // check shape
            if (!left.shape.CompareContents(right.shape)) {
                throw new ArgumentException($"Input tensors do not have same shape: {left.shape.ContentString()} vs {right.shape.ContentString()}");
            }
            if (!left.shape.CompareContents(output.shape)) {
                throw new ArgumentException($"Output tensor does not have same shape: Got: {output.shape.ContentString()} Expected: {right.shape.ContentString()}");
            }

            ComputeShader shader = Kernels.elementWiseBinary;

            ComputeBuffer leftBuffer = left.buffer;
            ComputeBuffer rightBuffer = right.buffer;
            ComputeBuffer outputBuffer = output.buffer;

            int kernelID = shader.FindKernel(kernelName);

            shader.SetBuffer(kernelID, Shader.PropertyToID("left"), leftBuffer);
            shader.SetBuffer(kernelID, Shader.PropertyToID("right"), rightBuffer);
            shader.SetBuffer(kernelID, Shader.PropertyToID("output"), outputBuffer);
            shader.SetInt(Shader.PropertyToID("count"), output.size);
            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = output.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }
        static void Call_Inplace(GPUTensorBuffer left, GPUTensorBuffer right, string kernelName) {
            // check shape
            if (!left.shape.CompareContents(right.shape)) {
                throw new System.ArgumentException($"Input tensors do not have same shape: {left.shape.ContentString()} vs {right.shape.ContentString()}");
            }

            ComputeShader shader = Kernels.elementWiseBinary;
            ComputeBuffer leftBuffer = left.buffer;
            ComputeBuffer rightBuffer = right.buffer;

            int kernelID = shader.FindKernel(kernelName);

            shader.SetBuffer(kernelID, Shader.PropertyToID("left"), leftBuffer);
            shader.SetBuffer(kernelID, Shader.PropertyToID("right"), rightBuffer);
            shader.SetInt(Shader.PropertyToID("count"), left.size);
            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = left.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }
        static void Call_Self(GPUTensorBuffer left, GPUTensorBuffer output, string kernelName) {
            // check shape
            if (!left.shape.CompareContents(output.shape)) {
                throw new System.ArgumentException($"Output tensor does not have same shape: Got: {output.shape.ContentString()} Expected: {left.shape.ContentString()}");
            }

            ComputeShader shader = Kernels.elementWiseBinary;
            ComputeBuffer leftBuffer = left.buffer;
            ComputeBuffer outputBuffer = output.buffer;

            int kernelID = shader.FindKernel(kernelName);

            shader.SetBuffer(kernelID, Shader.PropertyToID("left"), leftBuffer);
            shader.SetBuffer(kernelID, Shader.PropertyToID("output"), outputBuffer);
            shader.SetInt(Shader.PropertyToID("count"), output.size);
            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = output.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }
        static void Call_SelfInplace(GPUTensorBuffer left, string kernelName) {

            ComputeShader shader = Kernels.elementWiseBinary;
            ComputeBuffer leftBuffer = left.buffer;

            int kernelID = shader.FindKernel(kernelName);

            shader.SetBuffer(kernelID, Shader.PropertyToID("left"), leftBuffer);
            shader.SetInt(Shader.PropertyToID("count"), left.size);
            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = left.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }





        public static void Add(GPUTensorBuffer left, GPUTensorBuffer right, GPUTensorBuffer output) {
            const string name1 = "Add";
            const string name2 = name1 + _InplaceL;
            const string name3 = name1 + _InplaceR;
            const string name4 = name1 + _Self;
            const string name5 = name1 + _SelfInplace;

            Names n;
            n.normal = name1;
            n.inplaceL = name2;
            n.inplaceR = name3;
            n.self = name4;
            n.selfInplace = name5;

            Call(left, right, output, n);
        }
        public static void Multiply(GPUTensorBuffer left, GPUTensorBuffer right, GPUTensorBuffer output) {
            const string name1 = "Multiply";
            const string name2 = name1 + _InplaceL;
            const string name3 = name1 + _InplaceR;
            const string name4 = name1 + _Self;
            const string name5 = name1 + _SelfInplace;

            Names n;
            n.normal = name1;
            n.inplaceL = name2;
            n.inplaceR = name3;
            n.self = name4;
            n.selfInplace = name5;

            Call(left, right, output, n);
        }
        public static void Subtract(GPUTensorBuffer left, GPUTensorBuffer right, GPUTensorBuffer output) {
            const string name1 = "Subtract";
            const string name2 = name1 + _InplaceL;
            const string name3 = name1 + _InplaceR;
            const string name4 = name1 + _Self;
            const string name5 = name1 + _SelfInplace;

            Names n;
            n.normal = name1;
            n.inplaceL = name2;
            n.inplaceR = name3;
            n.self = name4;
            n.selfInplace = name5;

            Call(left, right, output, n);
        }
    }
}

