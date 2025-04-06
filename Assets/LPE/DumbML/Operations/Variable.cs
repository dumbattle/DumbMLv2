namespace DumbML {
    public class Variable : Operation {
        public ITensorBuffer buffer;
        public bool trainable;
        bool valueOutOfDate;

        FloatTensor _value;

        public FloatTensor value {
            get {
                if (valueOutOfDate) {
                    buffer.CopyTo(_value);
                    valueOutOfDate = false;
                }
                return _value;
            }
        }

        public Variable(params int[] shape) {
            foreach (var i in shape) {
                if (i < 0) {
                    throw new System.ArgumentException($"Variable must have fixed sized shapes. Got: {shape.ContentString()}");
                }
            }

            _value = new FloatTensor(shape);
            buffer = BLAS.Engine.GetTensorBuffer(DType.Float, shape);
            BLAS.Engine.Compute.SetTo0s(buffer);
            trainable = true;
            valueOutOfDate = false;

            BuildOp(shape, DType.Float);
        }

        public void InitValue(System.Func<float> initializer) {
            for (int i = 0; i < value.data.Length; i++) {
                _value.data[i] = initializer();
            }
            buffer.CopyFrom(_value);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            BLAS.Engine.Compute.Copy(buffer, result);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return null;
        }

        public void MarkBufferUpdated() {
            valueOutOfDate = true;
        }
    }
}
