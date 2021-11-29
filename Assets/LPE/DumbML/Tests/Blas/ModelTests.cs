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

            Tensor a = Tensor.FromArray(new[] { 1 });
            Tensor b = Tensor.FromArray(new[] { 1 });
            Tensor c = Tensor.FromArray(new[] { 1 });

            Tensor oa = new Tensor(op45.shape);
            Tensor ob = new Tensor(op56.shape);
            Tensor oc = new Tensor(op64.shape);

            m.Call(a, b, c).ToTensors(oa, ob, oc);

            Assert.True(oa[0] == 8);
            Assert.True(ob[0] == 8);
            Assert.True(oc[0] == 8);
            m.Dispose();
        }
    }
}