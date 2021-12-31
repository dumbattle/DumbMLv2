namespace DumbML {
    public class IntCPUTensorBuffer : CPUTensorBuffer<int> {
        public override DType dtype => DType.Int;

        public IntCPUTensorBuffer(params int[] shape) : base(shape) { }
    }
}