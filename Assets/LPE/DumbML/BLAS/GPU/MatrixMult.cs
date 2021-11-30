using UnityEngine;
using System;


namespace DumbML.BLAS.GPU {
    public static class MatrixMult {
        public static void Compute(GPUTensorBuffer l, GPUTensorBuffer r, GPUTensorBuffer dest) {
            var (mDim, innerDim, nDim) = CheckShapes(l, r, dest);

            ComputeBuffer leftBuffer = l.buffer;
            ComputeBuffer rightBuffer = r.buffer;
            ComputeBuffer outputBuffer = dest.buffer;

            ComputeShader shader = Kernels.matrixMult;
            int kernelID = shader.FindKernel("MatMult");

            shader.SetBuffer(kernelID, Shader.PropertyToID("left"), leftBuffer);
            shader.SetBuffer(kernelID, Shader.PropertyToID("right"), rightBuffer);
            shader.SetBuffer(kernelID, Shader.PropertyToID("output"), outputBuffer);

            shader.SetInts("lshape", l.shape);
            shader.SetInts("rshape", r.shape);
            shader.SetInts("oshape", dest.shape);

            shader.SetInt("lrank", l.Rank());
            shader.SetInt("rrank", r.Rank());
            shader.SetInt("orank", dest.Rank());

            shader.SetInt("mDim", mDim);
            shader.SetInt("innerDim", innerDim);
            shader.SetInt("nDim", nDim);

            shader.SetInt("count", dest.size);


            shader.GetKernelThreadGroupSizes(kernelID, out uint numThreads, out uint _, out uint _);
            int size = dest.size + (int)numThreads - 1;
            shader.Dispatch(kernelID, size / (int)numThreads, 1, 1);
        }

        private static (int, int, int) CheckShapes(GPUTensorBuffer l, GPUTensorBuffer r, GPUTensorBuffer dest) {
            int ldims = l.Rank();
            int rdims = r.Rank();
            int ddims = Mathf.Max(ldims, rdims);

            // check ranks > 2
            if (ldims < 2) {
                throw new ArgumentException($"MatrixMult requires tensors to have dimension of at least 2. Got shape: {l.shape.ContentString()}");
            }
            if (rdims < 2) {
                throw new ArgumentException($"MatrixMult requires tensors to have dimension of at least 2. Got shape: {r.shape.ContentString()}");
            }

            // dest has correct rank
            if (dest.Rank() != ddims) {
                throw new InvalidOperationException($"Output Tensors do not have correcct rank\n  Expected{ddims}\n  Got:{dest.shape.ContentString()}");
            }


            // check leading dimensions
            // determine number of batches
            int numBatches = 1;

            // can't start from 0 because l and r might have different ranks (ie. 1 of them might have implicit leading dimensions)
            // instead we use distancce from end to get dimension
            // negative = implied dimension of [1]
            // stop at 2 because we 2 dimensions are for matmult
            for (int i = ddims; i > 2; i--) {
                int dimSize = -1;

                int li = ldims - i;
                int ri = rdims - i;
                int di = ddims - i;

                int lsize = li >= 0 ? l.shape[li] : 1;
                int rsize = ri >= 0 ? r.shape[ri] : 1;

                // same
                if (rsize == lsize) {
                    dimSize = rsize;
                }
                // left is broadcastable to right
                else if (lsize == 1) {
                    dimSize = rsize;
                }

                // right is broadcastable to left
                else if (rsize == 1) {
                    dimSize = lsize;
                }

                // not compatable
                if (dimSize == -1) {
                    throw new InvalidOperationException(
                        $"Input Tensors do not have compatable leading dimensions for MatrixMult: {l.shape.ContentString()}, {r.shape.ContentString()}"
                    );
                }

                // dest doesnt have correct shape
                if (dimSize != dest.shape[di]) {
                    throw new InvalidOperationException(
                        $"Destination tensor does not have compatable batch dimensions: {dest.shape.ContentString()} Expected '{dimSize}' at index '{di}'"
                    );

                }

                numBatches *= dimSize;
            }

            // check shape compatability

            int lx = l.shape[ldims - 2];
            int ly = l.shape[ldims - 1];
            int rx = r.shape[rdims - 2];
            int ry = r.shape[rdims - 1];

            if (ly != rx) {
                throw new InvalidOperationException($"Tensors do not have compatible dimensions: {l.shape.ContentString()}, {r.shape.ContentString()}");
            }
            if (dest.shape[ddims - 2] != lx || dest.shape[ddims - 1] != ry) {
                throw new InvalidOperationException($"Output Tensor does not have correct shape - Expected: [ .., {lx}, {ry} ] Got: {dest.shape.ContentString()}");

            }

            return (lx, ly, ry);
        }
    }
}

