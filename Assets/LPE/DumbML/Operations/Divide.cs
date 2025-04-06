﻿using System.Collections.Generic;
using UnityEngine;

namespace DumbML {
    public class Divide : Operation {
        int[] shapeActual;

        public Divide(Operation a, Operation b) {
            shapeActual = OpUtility.GetBroadcastShape(a.shape, b.shape, shapeActual);
            BuildOp(shapeActual, DType.Float, a, b);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            shapeActual = OpUtility.GetBroadcastShape(inputs[0].shape, inputs[1].shape, shapeActual);
            result.SetShape(shapeActual);

            BLAS.Engine.Compute.Divide(inputs[0], inputs[1], result);
        }

        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            // reshape errors
            List<int> a = OpUtility.BroadcastBackwardsReductionShape(inputs[0].shape, error.shape);
            List<int> b = OpUtility.BroadcastBackwardsReductionShape(inputs[1].shape, error.shape);

            // reshape errors
            var agrad = a.Count > 0 ? new Reshape(new ReduceSum(error, inputs[0]), inputs[0]) : error;
            var bgrad = b.Count > 0 ? new Reshape(new ReduceSum(error, inputs[1]), inputs[1]) : error;
            // -ae/x^2

            Operation ae = new Multiply(inputs[0], bgrad);
            if (!ae.shape.CompareContents(bgrad.shape)) {
                ae = new ReduceSum(ae, bgrad);
            }
            ae = new Multiply(ae, -1);

            Operation bb = new Square(inputs[1]);

            bgrad = new Divide(ae, bb);

            return new Operation[] {
                new Divide(agrad, inputs[1]),
                bgrad
            };
        }
    }
}
