//namespace DumbML {
//    public class MatrixMult : Operation {
//        public MatrixMult(Operation a, Operation b) {
//            int aRank = a.shape.Length;
//            int bRank = b.shape.Length;

//            // Compatible matrix shapes
//            if (a.shape[aRank-1] != b.shape[bRank-2]) {
//                throw new System.ArgumentException($"Cannot MatrixMult tensors of shapes: {a.shape.ContentString()} by {b.shape.ContentString()}");
//            }

//            // Compatible batch dims
//            int[] shape = (int[])a.shape.Clone();
//            shape[shape.Length - 1] = b.shape[b.shape.Length - 1];
//            BuildOp(a.shape, a, b);
//        }


//        public override void Forward(ITensorBuffer[] inputs, ITensorBuffer result) {
//            throw new System.NotImplementedException();
//            BLAS.CPU.MaxtrixMult.Compute(inputs[0], inputs[1], result);
//        }


//        public override void Backward(ITensorBuffer[] inputs, ITensorBuffer output, ITensorBuffer error, ITensorBuffer[] results) {
//            throw new System.NotImplementedException();
//        }
//    }
//}
