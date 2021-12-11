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
            public static void Copy(ITensorBuffer src, ITensorBuffer dest) {
                Device deviceType = AssertSameDeviceType(src, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseSingle.Copy(AsFloatGPU(src), AsFloatGPU(dest));
                }
                else {
                    CPU.ElementWiseSingle.Copy(AsFloatCPU(src), AsFloatCPU(dest));
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
        }
    }

}
