using UnityEngine;

namespace DumbML {
    public class IntGPUTensorBuffer : GPUTensorBuffer {
        public override DType dtype => DType.Int;

        public IntGPUTensorBuffer(params int[] shape) : base(shape) { }

        protected override ComputeBuffer CreateNewBuffer(int count) {
            return new ComputeBuffer(count, sizeof(int));
        }
    }
}
