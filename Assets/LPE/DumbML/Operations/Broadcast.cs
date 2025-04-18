﻿using System.Collections.Generic;
using UnityEngine;


namespace DumbML {
    public class Broadcast : Operation {
        int[] targetShape;


        public Broadcast(Operation input, int[] targetShape) {
            this.targetShape = targetShape;
            BuildOp(targetShape, input.dtype, input);
        }

        public Broadcast(Operation input, Operation matchShape) {
            BuildOp(matchShape.shape, input.dtype, input, matchShape);
        }

        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
            if (inputs.Length == 1) {
                result.SetShape(targetShape);
                BLAS.Engine.Compute.Broadcast(inputs[0], targetShape, result);
            }
            else {
                result.SetShape(inputs[1].shape);
                try {
                    BLAS.Engine.Compute.Broadcast(inputs[0], inputs[1].shape, result);
                }
                catch (System.Exception) {
                    Debug.Log(id);
                    Debug.Log(inputs[0].shape.ContentString());
                    Debug.Log(inputs[1].shape.ContentString());
                    Debug.Log(result.shape.ContentString());
                    throw;
                }
            }
        }


        public override Operation[] BuildBackwards(Operation[] inputs, Operation output, Operation error) {
            List<int> broadcastedDims = new List<int>();

            for (int i = error.shape.Length; i > 0; i--) {
                int ii = inputs[0].shape.Length - i;
                int ei = error.shape.Length - i;

                bool isBroadcasted =
                    ii < 0 ||
                    (inputs[0].shape[ii] == 1 && error.shape[ei] != 1);
                if (isBroadcasted) {
                    broadcastedDims.Add(i);
                }
            }
            return new Operation[] {
                 new ReduceSum(error, broadcastedDims.ToArray())
            };
        }
    }
}
