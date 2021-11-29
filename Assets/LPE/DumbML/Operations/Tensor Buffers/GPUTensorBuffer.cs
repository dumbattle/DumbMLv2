using UnityEngine;

namespace DumbML {
    public class GPUTensorBuffer : ITensorBuffer {
        public int[] shape { get; private set; }
        public int size { get; private set; }
        public Device device => Device.gpu;


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
            buffer = new ComputeBuffer(size, sizeof(float));
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
            if (size > buffer.count) {
                buffer.Dispose();
                buffer = new ComputeBuffer(size, sizeof(float));
            }

        }

        public void CopyTo(CPUTensorBuffer dest) {
            dest.CopyFrom(this);
        }

        public void CopyFrom(CPUTensorBuffer src) {
            SetShape(src.shape);
      
            buffer.SetData(src.buffer, 0, 0, size);

        }


        public void CopyFrom(Tensor src) {
            SetShape(src.shape);

            buffer.SetData(src.data, 0, 0, size);
        }

        public void CopyTo(Tensor dest) {
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
