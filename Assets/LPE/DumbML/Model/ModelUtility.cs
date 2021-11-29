using System.Collections.Generic;
using System;

namespace DumbML {
    public static class ModelUtility {
        public static void ToTensors(this IReadOnlyList<ITensorBuffer> buffers, Tensor a) {
            if (buffers.Count < 1) {
                throw new ArgumentException($"List contains '{buffers.Count}' buffers. Trying to copy into {1} tensors.");
            }

            if (a != null) buffers[0].CopyTo(a);
        }

        public static void ToTensors(this IReadOnlyList<ITensorBuffer> buffers, Tensor a, Tensor b) {
            if (buffers.Count < 2) {
                throw new ArgumentException($"List contains '{buffers.Count}' buffers. Trying to copy into {2} tensors.");
            }

            if (a != null) buffers[0].CopyTo(a);
            if (b != null) buffers[1].CopyTo(b);
        }

        public static void ToTensors(this IReadOnlyList<ITensorBuffer> buffers, Tensor a, Tensor b, Tensor c) {
            if (buffers.Count < 3) {
                throw new ArgumentException($"List contains '{buffers.Count}' buffers. Trying to copy into {3} tensors.");
            }

            if (a != null) buffers[0].CopyTo(a);
            if (b != null) buffers[1].CopyTo(b);
            if (c != null) buffers[2].CopyTo(c);
        }

        public static void ToTensors(this IReadOnlyList<ITensorBuffer> buffers, Tensor a, Tensor b, Tensor c, Tensor d) {
            if (buffers.Count < 4) {
                throw new ArgumentException($"List contains '{buffers.Count}' buffers. Trying to copy into {4} tensors.");
            }

            if (a != null) buffers[0].CopyTo(a);
            if (b != null) buffers[1].CopyTo(b);
            if (c != null) buffers[2].CopyTo(c);
            if (d != null) buffers[3].CopyTo(d);
        }
    }
}
