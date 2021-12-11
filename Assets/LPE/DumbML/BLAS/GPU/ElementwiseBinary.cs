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
  
        static void Call(FloatGPUTensorBuffer left, FloatGPUTensorBuffer right, FloatGPUTensorBuffer output, Names names) {
            PartitionInfo pi = CheckShape(left.shape, right.shape, output.shape);
            ComputeShader shader = Kernels.elementWiseBinary;
            shader.SetInt(Shader.PropertyToID("lrank"), left.Rank());
            shader.SetInt(Shader.PropertyToID("rrank"), right.Rank());
            shader.SetInt(Shader.PropertyToID("drank"), output.Rank());
            shader.SetInt(Shader.PropertyToID("stride"), pi.stride);
            shader.SetInt(Shader.PropertyToID("batchCountL"), pi.lBatchCount);
            shader.SetInt(Shader.PropertyToID("batchCountR"), pi.rBatchCount);
            shader.SetInts(Shader.PropertyToID("lshape"), left.shape);
            shader.SetInts(Shader.PropertyToID("rshape"), right.shape);
            shader.SetInts(Shader.PropertyToID("oshape"), output.shape);

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
       
        
        static void Call_Normal(FloatGPUTensorBuffer left, FloatGPUTensorBuffer right, FloatGPUTensorBuffer output, string kernelName) {
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
        static void Call_Inplace(FloatGPUTensorBuffer left, FloatGPUTensorBuffer right, string kernelName) {
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
        static void Call_Self(FloatGPUTensorBuffer left, FloatGPUTensorBuffer output, string kernelName) {
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
        static void Call_SelfInplace(FloatGPUTensorBuffer left, string kernelName) {

            ComputeShader shader = Kernels.elementWiseBinary;
            ComputeBuffer leftBuffer = left.buffer;

            int kernelID = shader.FindKernel(kernelName);

            shader.SetBuffer(kernelID, Shader.PropertyToID("left"), leftBuffer);
            shader.SetInt(Shader.PropertyToID("count"), left.size);
            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = left.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }


        static PartitionInfo CheckShape(int[] l, int[] r, int[] d) {
            int threadSize = 10; // min array size each thread computes (don't want too many small threads) (arbitrarily set)


            int ldims = l.Length;
            int rdims = r.Length;
            int ddims = UnityEngine.Mathf.Max(ldims, rdims);

            if (ddims != d.Length) {
                throw new InvalidOperationException($"Output Tensors do not have correcct rank\n  Expected{ddims}\n  Got:{d.ContentString()}");
            }

            int strideSize = 1;
            int batchCount = 1;
            int batchCountL = 1;
            int batchCountR = 1;
            bool strideDone = false;

            for (int i = 1; i <= ddims; i++) {
                int dimSize = -1;

                int li = ldims - i;
                int ri = rdims - i;
                int di = ddims - i;

                int lsize = li >= 0 ? l[li] : 1;
                int rsize = ri >= 0 ? r[ri] : 1;

                // same
                if (rsize == lsize) {
                    dimSize = rsize;
                }
                // left is broadcastable to right
                else if (lsize == 1) {
                    dimSize = rsize;
                    strideDone = true;
                }
                // right is broadcastable to left
                else if (rsize == 1) {
                    dimSize = lsize;
                    strideDone = true;
                }

                // not compatable
                if (dimSize == -1) {
                    throw new InvalidOperationException(
                        $"Input Tensors do not have compatible dimensions: {l.ContentString()}, {r.ContentString()}"
                    );
                }

                // dest doesnt have correct shape
                if (dimSize != d[di]) {
                    throw new InvalidOperationException(
                        $"Destination tensor does not have compatable dimensions: {d.ContentString()} Expected '{dimSize}' at index '{di}'"
                    );
                }

                if (!strideDone && strideSize < threadSize) {
                    strideSize *= dimSize;
                    strideDone = true;
                }
                else {
                    batchCount *= dimSize;
                    batchCountL *= lsize;
                    batchCountR *= rsize;
                }
            }

            return new PartitionInfo() {
                batchCount = batchCount,
                lBatchCount = batchCountL,
                rBatchCount = batchCountR,
                stride = strideSize,
            };
        }


        struct PartitionInfo {
            public int batchCount;
            public int lBatchCount;
            public int rBatchCount;
            public int stride;
        }


        public static void Add(FloatGPUTensorBuffer left, FloatGPUTensorBuffer right, FloatGPUTensorBuffer output) {
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
        public static void Multiply(FloatGPUTensorBuffer left, FloatGPUTensorBuffer right, FloatGPUTensorBuffer output) {
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
        public static void Subtract(FloatGPUTensorBuffer left, FloatGPUTensorBuffer right, FloatGPUTensorBuffer output) {
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

