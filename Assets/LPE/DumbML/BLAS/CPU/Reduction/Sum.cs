namespace DumbML.BLAS.CPU {
    public static partial class Reduction {
        public static void Sum(FloatCPUTensorBuffer src, int[] axis, FloatCPUTensorBuffer dest) {
            Reduce<SumReducer>.Compute(src, axis, dest);
        }


        struct SumReducer : ReductionJob.IImplementation {
            float result;
            public void Start() {
                result = 0;
            }
            public void Next(float v) {
                result += v;
            }
            public float Complete() {
                return result;
            }

        }
    }
}