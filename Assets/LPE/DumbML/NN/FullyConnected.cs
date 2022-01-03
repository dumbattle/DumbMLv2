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
                weight.InitValue(() => UnityEngine.Random.Range(-.1f, .1f));
                if (useBias) {
                    bias = new Variable(outputSize);
                }
            }
            else {
                if (input.shape[input.shape.Length - 1] != weight.shape[0]) {
                    throw new System.ArgumentException($"Input tensor does not have compatible shape");
                }
            }
            var mm = new MatrixMult(input, weight);
            var ac = activation.Build(mm);
            var b = new Add(ac, bias);
            return b;
        }
    }
}

