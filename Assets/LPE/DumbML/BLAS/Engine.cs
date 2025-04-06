using UnityEngine;

namespace DumbML.BLAS {
    public static partial class Engine {
        public const int MAX_DIMENSION = 64;
        public static bool GPUEnabled;
        static bool GPUAvailable;
        static Device device => GPUEnabled && GPUAvailable ? Device.gpu : Device.cpu;

        static Engine() {
            GPUAvailable = SystemInfo.supportsComputeShaders;
            GPUEnabled = GPUAvailable;
        }


        public static ITensorBuffer GetTensorBuffer(DType type, params int[] shape) {
            switch (type) {
                case DType.Float:
                    if (device == Device.cpu) {
                        return new FloatCPUTensorBuffer(shape);
                    }
                    else if (device == Device.gpu) {
                        return new FloatGPUTensorBuffer(shape);
                    }
                    break;
                case DType.Int:
                    if (device == Device.cpu) {
                        return new IntCPUTensorBuffer(shape);
                    }
                    else if (device == Device.gpu) {
                        return new IntGPUTensorBuffer(shape);
                    }
                    break;
                case DType.Bool:
                    if (device == Device.cpu) {
                        return new BoolCPUTensorBuffer(shape);
                    }
                    else if (device == Device.gpu) {
                        return new BoolGPUTensorBuffer(shape);
                    }
                    break;
                default:
                    throw new System.NotImplementedException($"No buffer with type: {type}");
            }

            return null;
        }

        static Device AssertSameDeviceType(ITensorBuffer a, ITensorBuffer b) {
            Device result = a.device;

            if (b.device != result) {
                throw new System.ArgumentException($"Tensor buffers are on different devices: {a.device}, {b.device}");
            }

            return result;
        }
        static Device AssertSameDeviceType(ITensorBuffer a, ITensorBuffer b, ITensorBuffer c) {
            Device result = a.device;

            if (b.device != result) {
                throw new System.ArgumentException($"Tensor buffers are on different devices: {a.device}, {b.device}, {c.device}");
            }
            if (c.device != result) {
                throw new System.ArgumentException($"Tensor buffers are on different devices: {a.device}, {b.device}, {c.device}");
            }

            return result;
        }
    }

}
