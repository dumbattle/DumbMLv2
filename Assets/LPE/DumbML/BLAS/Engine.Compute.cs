namespace DumbML.BLAS {
    public static partial class Engine {
        public static class Compute {
            static FloatGPUTensorBuffer AsFloatGPU(ITensorBuffer src) {
                if (src is FloatGPUTensorBuffer result) {
                    return result;
                }
                throw new System.ArgumentException($"Expected float GPU tensor buffer\nGot{src.GetType()}");
            }
            static FloatCPUTensorBuffer AsFloatCPU(ITensorBuffer src) {
                if (src is FloatCPUTensorBuffer result) {
                    return result;
                }
                throw new System.ArgumentException($"Expected float CPU tensor buffer\nGot{src.GetType()}");
            }
            static BoolGPUTensorBuffer AsBoolGPU(ITensorBuffer src) {
                if (src is BoolGPUTensorBuffer result) {
                    return result;
                }
                throw new System.ArgumentException($"Expected bool GPU tensor buffer\nGot{src.GetType()}");
            }
            static BoolCPUTensorBuffer AsBoolCPU(ITensorBuffer src) {
                if (src is BoolCPUTensorBuffer result) {
                    return result;
                }
                throw new System.ArgumentException($"Expected bool CPU tensor buffer\nGot{src.GetType()}");
            }
            static IntGPUTensorBuffer AsIntGPU(ITensorBuffer src) {
                if (src is IntGPUTensorBuffer result) {
                    return result;
                }
                throw new System.ArgumentException($"Expected int GPU tensor buffer\nGot{src.GetType()}");
            }
            static IntCPUTensorBuffer AsIntCPU(ITensorBuffer src) {
                if (src is IntCPUTensorBuffer result) {
                    return result;
                }
                throw new System.ArgumentException($"Expected int CPU tensor buffer\nGot{src.GetType()}");
            }



            public static void Add(ITensorBuffer a, ITensorBuffer dest, float val) {
                Device deviceType = AssertSameDeviceType(a, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseSingleParam.Add(AsFloatGPU(a), AsFloatGPU(dest), val);
                }
                else {
                    CPU.ElementWiseFloatParam.Add(AsFloatCPU(a), AsFloatCPU(dest), val);
                }
            }
            public static void Add(ITensorBuffer a, ITensorBuffer b, ITensorBuffer dest) {
                Device deviceType = AssertSameDeviceType(a, b, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseBinary.Add(AsFloatGPU(a), AsFloatGPU(b), AsFloatGPU(dest));
                }
                else {
                    CPU.ElementwiseBinary.Add(AsFloatCPU(a), AsFloatCPU(b), AsFloatCPU(dest));
                }
            }
            public static void Broadcast(ITensorBuffer buffer, int[] targetShape, ITensorBuffer dest) {
                Device d = buffer.device;

                if (d == Device.gpu) {
                    GPU.Broadcast.Compute(AsFloatGPU(buffer), targetShape, AsFloatGPU(dest));
                }
                else {
                    CPU.Broadcast.Compute(AsFloatCPU(buffer), targetShape, AsFloatCPU(dest));
                }
            }

            public static void Cast(ITensorBuffer a, ITensorBuffer dest) {
                Device deviceType = AssertSameDeviceType(a, dest);

                if (deviceType == Device.gpu) {
                    switch ((atype: a.dtype, desttype: dest.dtype)) {
                        case (DType.Float, DType.Int):
                            GPU.Cast.FloatToInt(AsFloatGPU(a), AsIntGPU(dest));
                            return;
                        case (DType.Int, DType.Float):
                            GPU.Cast.IntToFloat(AsIntGPU(a), AsFloatGPU(dest));
                            return;
                        case (DType.Int, DType.Bool):
                            GPU.Cast.IntToBool(AsIntGPU(a), AsBoolGPU(dest));
                            return;
                        case (DType.Bool, DType.Int):
                            GPU.Cast.BoolToInt(AsBoolGPU(a), AsIntGPU(dest));
                            return;
                        case (DType.Bool, DType.Float):
                            GPU.Cast.BoolToFloat(AsBoolGPU(a), AsFloatGPU(dest));
                            return;
                        case (DType.Float, DType.Bool):
                            GPU.Cast.FloatToBool(AsFloatGPU(a), AsBoolGPU(dest));
                            return;
                        default:
                            throw new System.NotImplementedException("Cannot cast");
                    }

                }
                else {
                    switch ((atype: a.dtype, desttype: dest.dtype)) {
                        case (DType.Float, DType.Int):
                            CPU.Cast.Run(AsFloatCPU(a), AsIntCPU(dest));
                            return;
                        case (DType.Int, DType.Float):
                            CPU.Cast.Run(AsIntCPU(a), AsFloatCPU(dest));
                            return;
                        case (DType.Int, DType.Bool):
                            CPU.Cast.Run(AsIntCPU(a), AsBoolCPU(dest));
                            return;
                        case (DType.Bool, DType.Int):
                            CPU.Cast.Run(AsBoolCPU(a), AsIntCPU(dest));
                            return;
                        case (DType.Bool, DType.Float):
                            CPU.Cast.Run(AsBoolCPU(a), AsFloatCPU(dest));
                            return;
                        case (DType.Float, DType.Bool):
                            CPU.Cast.Run(AsFloatCPU(a), AsBoolCPU(dest));
                            return;
                        default:
                            throw new System.NotImplementedException("Cannot cast");

                    }
                }
            }
            public static void Copy(ITensorBuffer src, ITensorBuffer dest, bool ignoreShape = false) {
                Device deviceType = AssertSameDeviceType(src, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseSingle.Copy(AsFloatGPU(src), AsFloatGPU(dest), ignoreShape);
                }
                else {
                    CPU.ElementWiseSingle.Copy(AsFloatCPU(src), AsFloatCPU(dest), ignoreShape);
                }
            }
            public static void Clear(ITensorBuffer buffer) {
                Device d = buffer.device;

                if (d == Device.gpu) {
                    GPU.SetValues.Zero(AsFloatGPU(buffer));
                }
                else {
                    CPU.SetValues.Zero(AsFloatCPU(buffer));
                }
            }
            public static void ElementwiseEquals(ITensorBuffer a, ITensorBuffer b, ITensorBuffer dest) {
                Device deviceType = AssertSameDeviceType(a, b, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseBinary.Equals(AsFloatGPU(a), AsFloatGPU(b), AsBoolGPU(dest));
                }
                else {
                    CPU.ElementwiseBinary.Equals(AsFloatCPU(a), AsFloatCPU(b), AsBoolCPU(dest));
                }
            }
            public static void MatrixMult(ITensorBuffer a, ITensorBuffer b, ITensorBuffer dest, bool transA, bool transB) {
                Device deviceType = AssertSameDeviceType(a, b, dest);

                if (deviceType == Device.gpu) {
                    GPU.MatrixMult.Compute(AsFloatGPU(a), AsFloatGPU(b), AsFloatGPU(dest), transA, transB);
                }
                else {
                    CPU.MatrixMult.Compute(AsFloatCPU(a), AsFloatCPU(b), AsFloatCPU(dest), transA, transB);
                }
            }
            public static void Multiply(ITensorBuffer a, float val, ITensorBuffer dest) {
                Device deviceType = AssertSameDeviceType(a, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseSingleParam.Multiply(AsFloatGPU(a), AsFloatGPU(dest), val);
                }
                else {
                    CPU.ElementWiseFloatParam.Multiply(AsFloatCPU(a), AsFloatCPU(dest), val);
                }
            }
            public static void Multiply(ITensorBuffer a, ITensorBuffer b, ITensorBuffer dest) {
                Device deviceType = AssertSameDeviceType(a, b, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseBinary.Multiply(AsFloatGPU(a), AsFloatGPU(b), AsFloatGPU(dest));
                }
                else {
                    CPU.ElementwiseBinary.Multiply(AsFloatCPU(a), AsFloatCPU(b), AsFloatCPU(dest));
                }
            }
            public static void ReduceSum(ITensorBuffer buffer, int[] axis, ITensorBuffer dest) {
                Device d = buffer.device;

                if (d == Device.gpu) {
                    GPU.Reduction.Sum(AsFloatGPU(buffer), axis, AsFloatGPU(dest));
                }
                else {
                    CPU.Reduction.Sum(AsFloatCPU(buffer), axis, AsFloatCPU(dest));
                }
            }
            public static void ReLU(ITensorBuffer a, ITensorBuffer dest) {
                Device deviceType = AssertSameDeviceType(a, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseSingle.ReLU(AsFloatGPU(a), AsFloatGPU(dest));
                }
                else {
                    CPU.ElementWiseSingle.ReLU(AsFloatCPU(a), AsFloatCPU(dest));
                }
            }
            public static void SetTo0s(ITensorBuffer buffer) {
                Device d = buffer.device;

                if (d == Device.gpu) {
                    GPU.SetValues.Zero(AsFloatGPU(buffer));
                }
                else {
                    CPU.SetValues.Zero(AsFloatCPU(buffer));
                }
            }
            public static void SetTo1s(ITensorBuffer buffer) {
                Device d = buffer.device;

                if (d == Device.gpu) {
                    GPU.SetValues.One(AsFloatGPU(buffer));
                }
                else {
                    CPU.SetValues.One(AsFloatCPU(buffer));
                }
            }
            public static void Square(ITensorBuffer src, ITensorBuffer dest) {
                Device deviceType = AssertSameDeviceType(src, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseSingle.Sqr(AsFloatGPU(src), AsFloatGPU(dest));
                }
                else {
                    CPU.ElementWiseSingle.Sqr(AsFloatCPU(src), AsFloatCPU(dest));
                }
            }
            public static void Subtract(ITensorBuffer a, ITensorBuffer b, ITensorBuffer dest) {
                Device deviceType = AssertSameDeviceType(a, b, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseBinary.Subtract(AsFloatGPU(a), AsFloatGPU(b), AsFloatGPU(dest));
                }
                else {
                    CPU.ElementwiseBinary.Subtract(AsFloatCPU(a), AsFloatCPU(b), AsFloatCPU(dest));
                }
            }
        
            public static void Transpose(ITensorBuffer buffer, int[] perm, ITensorBuffer dest) {
                Device d = buffer.device;

                if (d == Device.gpu) {
                    GPU.Transpose.Compute(AsFloatGPU(buffer), perm, AsFloatGPU(dest));
                }
                else {
                    CPU.Transpose.Compute(AsFloatCPU(buffer), perm, AsFloatCPU(dest));
                }
            }
        }
    }

}
