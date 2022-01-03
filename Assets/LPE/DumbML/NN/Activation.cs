using System.Collections.Generic;

namespace DumbML.NN {
    public abstract class Activation {
        public static readonly Activation None = new _None();
        public static readonly Activation ReLU = new _ReLU();


        public abstract Operation Build(Operation input);


        class _None : Activation {
            public override Operation Build(Operation input) {
                return input;
            }
        }
        class _ReLU : Activation {
            public override Operation Build(Operation input) {
                return new ReLU(input);
            }
        }
    }

   public class TensorStackBuilder<T> {
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
            if (current == count) {
                throw new System.InvalidOperationException($"Reached end of stack");
            }
            int start = current * stride;

            System.Array.Copy(t.data, 0, destination.data, start, t.size);
            current++;
        }


        public Tensor<T> Complete() {
            return destination;
        }
    }

}

