using UnityEngine;

namespace DumbML {
    public class FloatGPUTensorBuffer : GPUTensorBuffer<float> {
        public override DType dtype => DType.Float;

        public FloatGPUTensorBuffer(params int[] shape) : base(shape) { }

        protected override ComputeBuffer CreateNewBuffer(int count) {
            return new ComputeBuffer(count, sizeof(float));
        }
    }
}
