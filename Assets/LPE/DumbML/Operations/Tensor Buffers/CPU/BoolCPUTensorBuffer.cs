namespace DumbML {
    public class BoolCPUTensorBuffer : CPUTensorBuffer<bool> {
        public override DType dtype => DType.Bool;

        public BoolCPUTensorBuffer(params int[] shape) : base(shape) { }
    }
}