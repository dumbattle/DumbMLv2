using System.Collections;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using UnityEngine;


namespace DumbML {
    public abstract class Operation {
        static int nextID;

        public int[] shape { get; private set; }
        public Operation[] inner;

        public string name { get; private set; }
        public DType dtype { get; private set; }
        public int id { get; private set; }
        public ITensorBuffer outputBuffer;

        protected void BuildOp(int[] shape, DType dtype, params Operation[] inner) {
            this.shape = (int[])shape.Clone();
            this.inner = (Operation[])inner.Clone();
            this.dtype = dtype;
            id = nextID;
            nextID++;
            var childrenLabel = (from x in inner select x.id).ToArray();

            //Debug.Log($"Op{id}({GetType().Name})[{shape.ContentString()}] from ({childrenLabel.ContentString()})");
           
        }


        public abstract void Forward(ITensorBuffer[] inputs, ITensorBuffer result);

        public abstract Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error);


        public void SetName(string name) {
            this.name = name;
        }
        public static implicit operator Operation(float f) {
            return new ConstantFloat(FloatTensor.FromArray(new float[] { f }));
        }


        public static Operation operator +(Operation l, Operation r) {
            return new Add(l, r);
        }
        public static Operation operator -(Operation l, Operation r) {
            return new Subtract(l, r);
        }
        public static Operation operator *(Operation l, Operation r) {
            return new Multiply(l, r);
        }
        public static Operation operator /(Operation l, Operation r) {
            return new Divide(l, r);
        }
        public static Operation operator -(Operation op) {
            return op * -1;
        }
    }
}
