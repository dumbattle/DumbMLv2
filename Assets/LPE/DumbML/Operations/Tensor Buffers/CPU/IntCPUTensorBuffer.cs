namespace DumbML {
    public class IntCPUTensorBuffer : CPUTensorBuffer<int> {
        public override DType dtype => DType.Float;

        public IntCPUTensorBuffer(params int[] shape) : base(shape) { }
    }
}