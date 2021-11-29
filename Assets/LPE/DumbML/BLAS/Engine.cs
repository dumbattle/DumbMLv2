using UnityEngine;

namespace DumbML.BLAS {
    public static partial class Engine {
        public static bool GPUEnabled;
        static Device device => GPUEnabled ? Device.gpu : Device.cpu;

        static Engine() {
            GPUEnabled = SystemInfo.supportsComputeShaders;
        }


        public static ITensorBuffer GetTensorBuffer(params int[] shape) {
            if (device == Device.cpu) {
                return new CPUTensorBuffer(shape);
            }
            else if (device == Device.gpu) {
                return new GPUTensorBuffer(shape);
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
