using System;


namespace DumbML {
    public class IntTensor : Tensor<int> {
        public override DType dtype => DType.Int;

        public IntTensor(params int[] shape) : base(shape) { }

        public static IntTensor FromArray(Array A) {
            int[] shape = new int[A.Rank];
            for (int i = 0; i < A.Rank; i++) {
                shape[i] = A.GetLength(i);
            }

            IntTensor result = new IntTensor(shape);
            int index = 0;

            foreach (var val in A) {
                result.data[index] = Convert.ToInt32(val);
                index++;
            }

            return result;
        }
    }
}