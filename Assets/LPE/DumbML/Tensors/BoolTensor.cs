using System;


namespace DumbML {
    public class BoolTensor : Tensor<bool> {
        public override DType dtype => DType.Bool;

        public BoolTensor(params int[] shape) : base(shape) { }

        public static BoolTensor FromArray(Array A) {
            int[] shape = new int[A.Rank];
            for (int i = 0; i < A.Rank; i++) {
                shape[i] = A.GetLength(i);
            }

            BoolTensor result = new BoolTensor(shape);
            int index = 0;

            foreach (var val in A) {
                result.data[index] = Convert.ToBoolean(val);
                index++;
            }

            return result;
        }
    }
}