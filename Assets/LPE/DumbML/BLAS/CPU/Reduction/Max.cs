namespace DumbML.BLAS.CPU {
    public static partial class Reduction {
        public static void Max(FloatCPUTensorBuffer src, int[] axis, FloatCPUTensorBuffer dest) {
            Reduce<MaxReducer>.Compute(src, axis, dest);
        }


        struct MaxReducer : ReductionJob.IImplementation {
            float result;

            public void Start() {
                result = float.NegativeInfinity;
            }
            public void Next(float v) {
                result = v > result ? v : result;
            }
            public float Complete() {
                return result;
            }

        }
    }
}