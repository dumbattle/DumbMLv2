using System.Collections;
using System.Collections.Generic;
using System;
using UnityEngine;


namespace DumbML {
    public class FloatTensor : Tensor<float> {
        public override DType dtype => DType.Float;

        public FloatTensor(params int[] shape) : base(shape) { }

        public static FloatTensor FromArray(Array A) {
            int[] shape = new int[A.Rank];
            for (int i = 0; i < A.Rank; i++) {
                shape[i] = A.GetLength(i);
            }

            FloatTensor result = new FloatTensor(shape);
            int index = 0;

            foreach (var val in A) {
                result.data[index] = Convert.ToSingle(val);
                index++;
            }
            return result;
        }
    }
}