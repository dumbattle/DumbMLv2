namespace DumbML {
    public static class TensorBufferExtensions {
        public static int Rank(this ITensorBuffer b) {
            return b.shape.Length;
        }
    }
}
