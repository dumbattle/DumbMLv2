namespace DumbML.NN {
    public class FullyConnected {
        int outputSize;
        bool useBias;
        Activation activation;

        Variable weight;
        Variable bias;

        public FullyConnected(int outputSize, Activation activation = null, bool useBias = true) {
            this.outputSize = outputSize;
            this.activation = activation ?? Activation.None;
            this.useBias = useBias;
        }


        public Operation Build(Operation input) {
            if (weight == null) {
                weight = new Variable(input.shape[input.shape.Length - 1], outputSize);
                weight.InitValue(() => UnityEngine.Random.Range(-.001f, .001f));
                if (useBias) {
                    bias = new Variable(outputSize);
                }
            }
            else {
                if (input.shape[input.shape.Length - 1] != weight.shape[0]) {
                    throw new System.ArgumentException($"Input tensor does not have compatible shape");
                }
            }
            Operation x = new MatrixMult(input, weight);
            if (useBias) {
                x = x + bias;
            }
            x = activation.Build(x);
            return x;
        }
    }
}

