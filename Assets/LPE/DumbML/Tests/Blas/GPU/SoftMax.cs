using NUnit.Framework;
using DumbML;
using UnityEngine;


namespace Tests.BLAS {
    namespace GPU {
        public class SoftMax {
            [Test]
            public void Simple() {
                Operation a = new ConstantFloat(FloatTensor.FromArray(new[,] { { 1, 2, 3} }));
                Operation sm = a.Softmax();

                FloatTensor output = new FloatTensor(1, 3);
                Model m = new Model(new InputOp[0], sm);
                m.Call().ToTensors(output);
                Debug.Log(output);

                m.Dispose();
            }
        }
    }

}