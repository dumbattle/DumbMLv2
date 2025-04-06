using UnityEngine;

namespace DumbML.NN {
    public abstract class TensorStackBuilder {
        public abstract void Begin(Tensor t);
        public abstract void Add(Tensor t);
        public abstract Tensor Finish();
        public abstract bool Done();

        public static TensorStackBuilder Get(DType t, int[] shape) {
            switch (t) {
                case DType.Float:
                    return new TensorStackBuilder<float>(shape);
                case DType.Int:
                    return new TensorStackBuilder<int>(shape);
                case DType.Bool:
                    return new TensorStackBuilder<bool>(shape);
            }
            return null;
        }
    }
    public class TensorStackBuilder<T> : TensorStackBuilder {
        int[] dataShape;
        Tensor<T> destination;
        int stride;
        int count;
        int current;

        public TensorStackBuilder(int[] dataShape) {
            this.dataShape = (int[])dataShape.Clone();
            for (int i = 0; i < dataShape.Length; i++) {
                stride *= dataShape[i];
            }
        }
        public override void Begin(Tensor t) {
            if (t is Tensor<T> tt) {
                Begin(tt);
            }
        }
        public override void Add(Tensor t) {
            if (t is Tensor<T> tt) {
                Add(tt);
            }
        }
        public override Tensor Finish() {
            return Complete();
        }
        public override bool Done() {
            return current == count;
        }
        public void Begin(Tensor<T> destination) {
            this.destination = destination;
            count = 1;

            for (int i = 0; i < destination.shape.Length; i++) {
                int ind = destination.shape.Length - i - 1;

                if (i < dataShape.Length) {
                    int dind = dataShape.Length - i - 1;
                    if (dataShape[dind] != destination.shape[ind]) {
                        throw new System.ArgumentException($"Wrong Tensor stack shape");
                    }
                }

                count *= destination.shape[i];
            }

            current = 0;
            System.Array.Clear(destination.data, 0, destination.size);
        }

        public void Add(Tensor<T> t) {
            if (Done()) {
                throw new System.InvalidOperationException($"Reached end of stack");
            }
            int start = current;

            System.Array.Copy(t.data, 0, destination.data, start, t.size);
            current += t.size;
        }


        public Tensor<T> Complete() {
            return destination;
        }
    }

}

