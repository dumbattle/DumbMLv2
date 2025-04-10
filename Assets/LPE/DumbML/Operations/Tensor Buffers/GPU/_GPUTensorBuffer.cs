﻿using UnityEngine;

namespace DumbML {
    public abstract class GPUTensorBuffer<T> : ITensorBuffer where T : struct {
        public int[] shape { get; private set; }
        public int size { get; private set; }
        public int capacity => buffer.count;
        public Device device => Device.gpu;
        public abstract DType dtype { get; }


        public ComputeBuffer buffer;



        public GPUTensorBuffer(params int[] shape) {
            this.shape = new int[shape.Length];

            size = 1;

            for (int i = 0; i < shape.Length; i++) {
                int s = Mathf.Abs(shape[i]);
                this.shape[i] = s;
                size *= s;
            }
            buffer = CreateNewBuffer(size);
        }

        protected abstract ComputeBuffer CreateNewBuffer(int count);


        public void SetShape(int[] shape) {
            if (shape.Length != this.shape.Length) {
                this.shape = new int[shape.Length];
            }


            // set shape and size
            size = 1;
            for (int i = 0; i < shape.Length; i++) {
                this.shape[i] = shape[i];
                size *= shape[i];
            }

            // resize buffer if neccessary
            if (size > buffer.count) {
                buffer.Dispose();
                buffer = CreateNewBuffer(size);
            }
        }


        /// <summary>
        /// for some ops, it is neccessary to use a larger buffer than necessary in order to make computations easier
        /// </summary>
        public bool ExpandBuffer(int size) {
            if (size <= this.size) {
                return false;
            }

            buffer.Dispose();
            buffer = CreateNewBuffer(size);
            return true;
        }


        public void CopyFrom(Tensor src) {
            if (src is Tensor<T> t) {
                CopyFrom(t);
            }
            else {
                throw new System.ArgumentException($"Invalid dtpyes:\nSrc: {src.dtype}\nDest: {dtype}");
            }
        }

        public void CopyTo(Tensor dest) {
            if (dest is Tensor<T> t) {
                CopyTo(t);
            }
            else {
                throw new System.ArgumentException($"Invalid dtpyes:\nSrc: {dtype}\nDest: {dest.dtype}");
            }
        }

        protected void CopyFrom(Tensor<T> src) {
            SetShape(src.shape);

            buffer.SetData(src.data, 0, 0, size);
        }

        protected void CopyTo(Tensor<T> dest) {
            if (!ShapeUtility.SameShape(shape, dest.shape)) {
                throw new System.ArgumentException($"Destination tensor does not have correct shape. Expected: {shape.ContentString()} Got: {dest.shape.ContentString()}");
            }
            buffer.GetData(dest.data, 0, 0, size);
        }
        public void Dispose() {
            buffer.Dispose();

            // set all other values to invalid values to discourage use after disposing
            shape = null;
            size = -1;
            buffer = null;
        }

    }
}
