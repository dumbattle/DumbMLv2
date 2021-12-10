using UnityEngine;
using System;


namespace DumbML {
    public abstract class CPUTensorBuffer<T> : ITensorBuffer {
        public int[] shape { get; private set; }
        public int size { get; private set; }
        public Device device => Device.cpu;
        public abstract DType dtype { get; }

        public T[] buffer;

        int[] shapeConstraints;


        public CPUTensorBuffer(params int[] shape) {
            this.shape = new int[shape.Length];
            shapeConstraints = (int[])shape.Clone();

            size = 1;

            for (int i = 0; i < shape.Length; i++) {
                int s = Mathf.Abs(shape[i]);
                this.shape[i] = s;
                size *= s;
            }

            buffer = new T[size];
        }
        public void SetShape(int[] shape) {
            // check valid shape
            if (shape.Length != shapeConstraints.Length) {
                throw new System.ArgumentException($"Desired shape ({shape.ContentString()}) is does not meet constraints ({this.shape.ContentString()})");
            }

            for (int i = 0; i < shape.Length; i++) {
                int c = this.shape[i];
                int s = shape[i];

                if (s < 0) {

                    throw new System.ArgumentException($"Invalid shape ({shape.ContentString()})");
                }
                if (c >= 0 && c != shape[i]) {
                    throw new System.ArgumentException($"Desired shape ({shape.ContentString()}) is does not meet constraints ({this.shape.ContentString()})");
                }

            }

            // set shape and size
            size = 1;
            for (int i = 0; i < shape.Length; i++) {
                this.shape[i] = shape[i];
                size *= shape[i];
            }

            // resize buffer if neccessary
            if (size > buffer.Length) {
                buffer = new T[size];
            }

        }


        public void CopyFrom<U>(Tensor<U> src) {
            if (src is Tensor<T> t) {
                CopyFrom(t);
            }
            else {
                throw new ArgumentException($"Invalid dtpyes:\nSrc: {src.dtype}\nDest: {dtype}");
            }
        }

        public void CopyTo<U>(Tensor<U> dest) {
            if (dest is Tensor<T> t) {
                CopyTo(t);
            }
            else {
                throw new ArgumentException($"Invalid dtpyes:\nSrc: {dtype}\nDest: {dest.dtype}");
            }
        }

        protected void CopyFrom(Tensor<T> src) {
            SetShape(src.shape);

            for (int i = 0; i < size; i++) {

                buffer[i] = src.data[i];
            }
        }

        protected void CopyTo(Tensor<T> dest) {
            if (!ShapeUtility.SameShape(shape, dest.shape)) {
                throw new System.ArgumentException($"Destination tensor does not have correct shape. Expected: {shape.ContentString()} Got: {dest.shape.ContentString()}");
            }
            for (int i = 0; i < size; i++) {
                dest.data[i] = buffer[i];
            }
        }

        public void Dispose() {
            // set all values to invalid values to discourage use after disposing
            buffer = null;
            shape = null;
            size = -1;
            buffer = null;
            shapeConstraints = null;
        }
    }
}