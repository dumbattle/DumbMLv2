using System.Collections.Generic;
using System;

namespace DumbML {
    public static class ModelUtility {
        public static void ToTensors<A>(this IReadOnlyList<ITensorBuffer> buffers, Tensor<A> a) {
            if (buffers.Count < 1) {
                throw new ArgumentException($"List contains '{buffers.Count}' buffers. Trying to copy into {1} tensors.");
            }

            if (a != null) buffers[0].CopyTo(a);
        }

        public static void ToTensors<A, B>(this IReadOnlyList<ITensorBuffer> buffers, Tensor<A> a, Tensor<B> b) {
            if (buffers.Count < 2) {
                throw new ArgumentException($"List contains '{buffers.Count}' buffers. Trying to copy into {2} tensors.");
            }

            if (a != null) buffers[0].CopyTo(a);
            if (b != null) buffers[1].CopyTo(b);
        }

        public static void ToTensors<A, B, C>(this IReadOnlyList<ITensorBuffer> buffers, Tensor<A> a, Tensor<B> b, Tensor<C> c) {
            if (buffers.Count < 3) {
                throw new ArgumentException($"List contains '{buffers.Count}' buffers. Trying to copy into {3} tensors.");
            }

            if (a != null) buffers[0].CopyTo(a);
            if (b != null) buffers[1].CopyTo(b);
            if (c != null) buffers[2].CopyTo(c);
        }

        public static void ToTensors<A, B, C, D>(this IReadOnlyList<ITensorBuffer> buffers, Tensor<A> a, Tensor<B> b, Tensor<C> c, Tensor<D> d) {
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
