namespace DumbML.BLAS {
    public static partial class Engine {
        public static class Compute {
            public static void Add(ITensorBuffer a, ITensorBuffer dest, float val) {
                Device deviceType = AssertSameDeviceType(a, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseSingleParam.Add((GPUTensorBuffer)a, (GPUTensorBuffer)dest, val);
                }
                else {
                    CPU.ElementWiseFloatParam.Add((CPUTensorBuffer)a, (CPUTensorBuffer)dest, val);
                }
            }
            public static void Add(ITensorBuffer a, ITensorBuffer b, ITensorBuffer dest) {
                Device deviceType = AssertSameDeviceType(a, b, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseBinary.Add((GPUTensorBuffer)a, (GPUTensorBuffer)b, (GPUTensorBuffer)dest);
                }
                else {
                    CPU.ElementwiseBinary.Add((CPUTensorBuffer)a, (CPUTensorBuffer)b, (CPUTensorBuffer)dest);
                }
            }
            public static void Copy(ITensorBuffer src, ITensorBuffer dest) {
                Device deviceType = AssertSameDeviceType(src, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseSingle.Copy((GPUTensorBuffer)src, (GPUTensorBuffer)dest);
                }
                else {
                    CPU.ElementWiseSingle.Copy((CPUTensorBuffer)src, (CPUTensorBuffer)dest);
                }
            }

            public static void Clear(ITensorBuffer buffer) {
                Device d = buffer.device;

                if (d == Device.gpu) {
                    GPU.SetValues.Zero((GPUTensorBuffer)buffer);
                }
                else {
                    CPU.SetValues.Zero((CPUTensorBuffer)buffer);
                }
            }


            public static void Multiply(ITensorBuffer a, float val, ITensorBuffer dest) {
                Device deviceType = AssertSameDeviceType(a, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseSingleParam.Multiply((GPUTensorBuffer)a, (GPUTensorBuffer)dest, val);
                }
                else {
                    CPU.ElementWiseFloatParam.Multiply((CPUTensorBuffer)a, (CPUTensorBuffer)dest, val);
                }
            }
            public static void Multiply(ITensorBuffer a, ITensorBuffer b, ITensorBuffer dest) {
                Device deviceType = AssertSameDeviceType(a, b, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseBinary.Multiply((GPUTensorBuffer)a, (GPUTensorBuffer)b, (GPUTensorBuffer)dest);
                }
                else {
                    CPU.ElementwiseBinary.Multiply((CPUTensorBuffer)a, (CPUTensorBuffer)b, (CPUTensorBuffer)dest);
                }
            }
            public static void SetTo1s(ITensorBuffer buffer) {
                Device d = buffer.device;

                if (d == Device.gpu) {
                    GPU.SetValues.One((GPUTensorBuffer)buffer);
                }
                else {
                    CPU.SetValues.One((CPUTensorBuffer)buffer);
                }
            }
            public static void Square(ITensorBuffer src, ITensorBuffer dest) {
                Device deviceType = AssertSameDeviceType(src, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseSingle.Sqr((GPUTensorBuffer)src, (GPUTensorBuffer)dest);
                }
                else {
                    CPU.ElementWiseSingle.Sqr((CPUTensorBuffer)src, (CPUTensorBuffer)dest);
                }
            }
            public static void Subtract(ITensorBuffer a, ITensorBuffer b, ITensorBuffer dest) {
                Device deviceType = AssertSameDeviceType(a, b, dest);

                if (deviceType == Device.gpu) {
                    GPU.ElementwiseBinary.Subtract((GPUTensorBuffer)a, (GPUTensorBuffer)b, (GPUTensorBuffer)dest);
                }
                else {
                    CPU.ElementwiseBinary.Subtract((CPUTensorBuffer)a, (CPUTensorBuffer)b, (CPUTensorBuffer)dest);
                }
            }
            //public static void Add(ITensorBuffer a, ITensorBuffer b, ITensorBuffer d) {
            //    if (GPUEnabled) {
            //        GPU.ElementwiseBinary.Add(a.GetGPUTensor(), b.GetGPUTensor(), d.GetGPUTensor());
            //    }
            //    else {
            //        CPU.ElementwiseBinary.Add(a.GetCPUTensor(), b.GetCPUTensor(), d.GetCPUTensor());
            //    }
            //}
            //public static void MatrixMult2x2(ITensorBuffer l, ITensorBuffer r, ITensorBuffer dest) {
            //    if (GPUEnabled) {

            //    }
            //    else {
            //        CPU.MaxtrixMult2x2.Compute(l.GetCPUTensor(), r.GetCPUTensor(), dest.GetCPUTensor());
            //    }
            //}
            //public static void MatrixMult2x2Backwards(ITensorBuffer l, ITensorBuffer r, ITensorBuffer e, ITensorBuffer le, ITensorBuffer re) {
            //    if (GPUEnabled) {

            //    }
            //    else {
            //        CPU.MaxtrixMult2x2.Backwards(l.GetCPUTensor(), r.GetCPUTensor(), e.GetCPUTensor(), le.GetCPUTensor(), re.GetCPUTensor());
            //    }
            //}
            //public static void ClearData(ITensorBuffer t) {

            //}
            //public static void CopyValues(ITensorBuffer src, ITensorBuffer dest) {

            //}
        }
    }

}
