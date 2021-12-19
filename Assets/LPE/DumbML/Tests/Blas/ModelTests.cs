using NUnit.Framework;
using DumbML;
using System.Runtime.InteropServices;
using System.Diagnostics;
using System.Collections.Generic;

namespace Tests.DumbMLTests {
    public class ModelTests {
        [Test]
        public void Order() {
            int[] shape = { 1 };

            InputOp inA = new InputOp(shape);
            InputOp inB = new InputOp(shape);
            InputOp inC = new InputOp(shape);

            Operation opAB = new Add(inA, inB);
            Operation opBC = new Add(inB, inC);
            Operation opCA = new Add(inC, inA);

            Operation op12 = new Add(opAB, opBC);
            Operation op23 = new Add(opBC, opCA);
            Operation op31 = new Add(opCA, opAB);

            Operation op45 = new Add(op12, op23);
            Operation op56 = new Add(op23, op31);
            Operation op64 = new Add(op31, op12);


            Model m = new Model(new[] { inA, inB, inC }, new[] { op45, op56, op64 });

            FloatTensor a = FloatTensor.FromArray(new[] { 1 });
            FloatTensor b = FloatTensor.FromArray(new[] { 1 });
            FloatTensor c = FloatTensor.FromArray(new[] { 1 });

            FloatTensor oa = new FloatTensor(op45.shape);
            FloatTensor ob = new FloatTensor(op56.shape);
            FloatTensor oc = new FloatTensor(op64.shape);

            m.Call(a, b, c).ToTensors(oa, ob, oc);

            Assert.True(oa[0] == 8);
            Assert.True(ob[0] == 8);
            Assert.True(oc[0] == 8);
            m.Dispose();
        }




        //[Test]
        //public void BackwardsModel() {
        //    Operation a = 3;
        //    Operation x = 2;

        //    Operation op = new Multiply(a, x);

        //    Model m = new Model(new InputOp[0], new[] { op });
        //    Model backwards = m.CreateBackwardsModel(new[] { x }, new[] { op });

        //    FloatTensor t = new FloatTensor(1);

        //    m.Call().ToTensors(t);
        //    UnityEngine.Debug.Log(t[0]);

        //    backwards.Call().ToTensors(t);
        //    UnityEngine.Debug.Log(t[0]);
            

        //    m.Dispose();
        //    backwards.Dispose();
        //}
    }
    public class OpUtilTests {
        [Test] 
        public void BroadCast() {
            int[] a = { -1, 24 };
            int[] b = { 1 };

            var r = OpUtility.GetBroadcastShape(a, b, null);
            CollectionAssert.AreEqual(r, new int[] { -1, 24});
        }
    }

}