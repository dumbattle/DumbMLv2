namespace DumbML.BLAS.CPU {
    public static partial class Reduction {
        public static void Sum(FloatCPUTensorBuffer src, int[] axis, FloatCPUTensorBuffer dest) {
            Reduce<SumReducer>.Compute(src, axis, dest);
        }


        class SumReducer : Reducer {
            float result;

            public override void Reset() {
                result = 0;
            }
            public override void Update(float val) {
                result += val;
            }
            public override float Complete() {
                return result;
            }
        }
    }
}