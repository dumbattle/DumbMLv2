namespace DumbML {
    public class FloatCPUTensorBuffer : CPUTensorBuffer<float> {
        public override DType dtype => DType.Float;

        public FloatCPUTensorBuffer(params int[] shape) : base(shape) { }
    }
}