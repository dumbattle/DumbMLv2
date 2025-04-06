using System.Collections.Generic;

namespace DumbML.NN {
    public abstract class Activation {
        public static readonly Activation None = new _None();
        public static readonly Activation ReLU = new _ReLU();
        public static readonly Activation Sigmoid = new _Sigmoid();


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
        class _Sigmoid : Activation {
            public override Operation Build(Operation input) {
                var result = input.Sigmoid();

                return result;
            }
        }
    }

}

