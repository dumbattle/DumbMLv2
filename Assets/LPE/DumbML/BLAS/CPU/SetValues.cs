namespace DumbML.BLAS.CPU {
    public static class SetValues {
        public static void One(CPUTensorBuffer input) {
            for (int i = 0; i < input.size; i++) {
                input.buffer[i] = 1;
            }
        }
        public static void Zero(CPUTensorBuffer input) {
            for (int i = 0; i < input.size; i++) {
                input.buffer[i] = 0;
            }
        }
    }
}