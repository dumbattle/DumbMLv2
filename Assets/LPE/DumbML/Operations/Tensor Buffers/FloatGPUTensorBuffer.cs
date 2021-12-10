using UnityEngine;

namespace DumbML {
    public class FloatGPUTensorBuffer : GPUTensorBuffer {
        public override DType dtype => DType.Float;

        public FloatGPUTensorBuffer(params int[] shape) : base(shape) { }

        protected override ComputeBuffer CreateNewBuffer(int count) {
            return new ComputeBuffer(size, sizeof(float));
        }
    }
}
