using UnityEngine;

namespace DumbML {
    public static class Math {
        public static Operation TEST_OP;

        public static Operation Softmax(this Operation op, int axis = -1) {
            Operation max = new AddDims(new ReduceMax(op, axis), new[] { axis });
            max = new NoGrad(max);
            op = op - max;

            Operation e = new Exp(op);

            Operation sum = new ReduceSum(e, axis);
            TEST_OP = sum;
            sum = new AddDims(sum, new[] { axis });
            var result = e / sum;

            return result;
        }

        public static Operation Sigmoid(this Operation op) {
            var ex = new Exp(op);
            return ex / (1 + ex);
        }
    }
}
