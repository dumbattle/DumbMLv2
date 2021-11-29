using System.Collections;
using System.Collections.Generic;
using UnityEngine;


namespace DumbML {
    public abstract class Operation {
        public int[] shape { get; private set; }
        public Operation[] inner;

        public string name { get; private set; }

        protected void BuildOp(int[] shape, params Operation[] inner) {
            this.shape = (int[])shape.Clone();
            this.inner = (Operation[])inner.Clone();


        }


        public abstract void Forward(ITensorBuffer[] inputs, ITensorBuffer result);
        public abstract void Backward(ITensorBuffer[] inputs, ITensorBuffer output, ITensorBuffer error, ITensorBuffer[] results);

        public void SetName(string name) {
            this.name = name;
        }
    }
}
