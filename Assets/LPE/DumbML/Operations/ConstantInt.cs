namespace DumbML {
    public class ConstantInt : Operation {
        public IntTensor value { get; private set; }
        public bool trainable;

        public ConstantInt(IntTensor src) {
            value = new IntTensor(src.shape);
            System.Array.Copy(src.data, value.data, src.size);
            BuildOp(src.shape, DType.Float);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            result.CopyFrom(value);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            return null;
        }
    }


}
