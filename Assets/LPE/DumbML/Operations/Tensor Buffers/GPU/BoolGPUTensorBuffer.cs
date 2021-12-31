using UnityEngine;

namespace DumbML {
    public class BoolGPUTensorBuffer : GPUTensorBuffer {
        public override DType dtype => DType.Bool;

        public BoolGPUTensorBuffer(params int[] shape) : base(shape) { }

        protected override ComputeBuffer CreateNewBuffer(int count) {
            return new ComputeBuffer(count, sizeof(bool));
        }
    }
}
