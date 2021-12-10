namespace DumbML {
    public static class TensorBufferExtensions {
        public static int Rank(this ITensorBuffer b) {
            return b.shape.Length;
        }
        public static void CopyTo(this FloatGPUTensorBuffer src, FloatCPUTensorBuffer dest) {
            dest.CopyFrom(src);
        }

        public static void CopyFrom(this FloatGPUTensorBuffer dest, FloatCPUTensorBuffer src) {
            dest.SetShape(src.shape);

            dest.buffer.SetData(src.buffer, 0, 0, dest.size);
        }

        public static void CopyFrom(this FloatCPUTensorBuffer dest, FloatGPUTensorBuffer src) {
            dest.SetShape(src.shape);
            src.buffer.GetData(dest.buffer);
        }
        public static void CopyTo(this FloatCPUTensorBuffer src, FloatGPUTensorBuffer dest) {
            dest.CopyFrom(src);
        }

    }
}
