using UnityEngine;

namespace DumbML {
    public abstract class GPUTensorBuffer : ITensorBuffer {
        public int[] shape { get; private set; }
        public int size { get; private set; }
        public Device device => Device.gpu;
        public abstract DType dtype { get; }


        public ComputeBuffer buffer;

        int[] shapeConstraints;


        public GPUTensorBuffer(params int[] shape) {
            this.shape = new int[shape.Length];
            shapeConstraints = (int[])shape.Clone();

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
            if (size > buffer.count) {
                buffer.Dispose();
                buffer = CreateNewBuffer(size);
            }
        }

        public void CopyFrom<T>(Tensor<T> src) {
            if (src.dtype != dtype) {
                throw new System.ArgumentException($"Invalid dtpyes:\nSrc: {src.dtype}\nDest: {dtype}");
            }
            SetShape(src.shape);

            buffer.SetData(src.data, 0, 0, size);
        }

        public void CopyTo<T>(Tensor<T> dest) {
            if (dest.dtype != dtype) {
                throw new System.ArgumentException($"Invalid dtpyes:\nSrc: {dtype}\nDest: {dest.dtype}");
            }
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
            shapeConstraints = null;
        }

    }
}
