using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace DumbML {
    public abstract class Operation {
        public int[] shape { get; private set; }
        public Operation[] inner;

        public string name { get; private set; }
        public DType dtype { get; private set; }


        protected void BuildOp(int[] shape, DType dtype, params Operation[] inner) {
            this.shape = (int[])shape.Clone();
            this.inner = (Operation[])inner.Clone();
            this.dtype = dtype;

        }


        public abstract void Forward(ITensorBuffer[] inputs, ITensorBuffer result);
        public abstract void Backward(ITensorBuffer[] inputs, ITensorBuffer output, ITensorBuffer error, ITensorBuffer[] results);

        public abstract Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error);


        public void SetName(string name) {
            this.name = name;
        }
        public static implicit operator Operation(float f) {
            return new ConstantFloat(FloatTensor.FromArray(new float[] { f }));
        }
    }
}
