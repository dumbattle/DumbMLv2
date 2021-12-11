﻿namespace DumbML {
    public class ConstantFloat : Operation {
        public FloatTensor value { get; private set; }
        public bool trainable;

        public ConstantFloat(FloatTensor src) {
            System.Array.Copy(src.data, value.data, src.size);
            BuildOp(src.shape, DType.Float);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            result.CopyFrom(value);
        }
        public override void Backward(ITensorBuffer[] inputs, ITensorBuffer output, ITensorBuffer error, ITensorBuffer[] results) { }
    }

}
