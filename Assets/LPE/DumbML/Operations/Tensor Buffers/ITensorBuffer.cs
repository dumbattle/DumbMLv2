﻿using System.Collections.Generic;

namespace DumbML {

    public interface ITensorBuffer : System.IDisposable {
        int[] shape { get; }
        int size { get; }
        int capacity { get; }
        Device device { get; }
        DType dtype{ get; }

        void SetShape(int[] shape);

        void CopyFrom(Tensor t);
        void CopyTo(Tensor t);
    }
}
