namespace DumbML {
    public class Variable : Operation {
        public FloatTensor value { get; private set; }
        public bool trainable;

        public Variable(params int[] shape) {
            foreach (var i in shape) {
                if (i < 0) {
                    throw new System.ArgumentException($"Variable must have fixed sized shapes. Got: {shape.ContentString()}");
                }
            }
            BuildOp(shape, DType.Float);
            value = new FloatTensor(shape);
            trainable = true;
        }

        public void InitValue(System.Func<float> initializer) {
            for (int i = 0; i < value.data.Length; i++) {
                value.data[i] = initializer();
            }
        }


        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            result.CopyFrom(value);
        }
        public override void Backward(ITensorBuffer[] inputs, ITensorBuffer output, ITensorBuffer error, ITensorBuffer[] results) { }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return null;
        }
    }
}
