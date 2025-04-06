using UnityEngine;

namespace DumbML.BLAS.GPU {
    /// <summary>
    /// Lazy-loaded ComputeShaders
    /// </summary>
    public static class Kernels {
        public static ComputeShader broadcast => _broadcast = _broadcast ?? Resources.Load<ComputeShader>("GPU Kernels/Broadcast");
        static ComputeShader _broadcast;

        public static ComputeShader cast => _cast = _cast ?? Resources.Load<ComputeShader>("GPU Kernels/Cast");
        static ComputeShader _cast;

        public static ComputeShader elementWiseBinary => _elementWiseBinary = _elementWiseBinary ?? Resources.Load<ComputeShader>("GPU Kernels/Elementwise Binary");
        static ComputeShader _elementWiseBinary;

        public static ComputeShader elementWiseSingleParam => _elementWiseSingleParam = _elementWiseSingleParam ?? Resources.Load<ComputeShader>("GPU Kernels/Elementwise Single With Param");
        static ComputeShader _elementWiseSingleParam;

        public static ComputeShader elementWiseSingle => _elementWiseSingle = _elementWiseSingle ?? Resources.Load<ComputeShader>("GPU Kernels/Elementwise Single");
        static ComputeShader _elementWiseSingle;

        public static ComputeShader matrixMult => _matrixMult = _matrixMult ?? Resources.Load<ComputeShader>("GPU Kernels/Matrix Mult");
        static ComputeShader _matrixMult;

        public static ComputeShader oneHot => _oneHot = _oneHot ?? Resources.Load<ComputeShader>("GPU Kernels/OneHot");
        static ComputeShader _oneHot;

        public static ComputeShader reduction => _reduction = _reduction ?? Resources.Load<ComputeShader>("GPU Kernels/Reduction");
        static ComputeShader _reduction;

        public static ComputeShader sampleCategorical => _sampleCategorical = _sampleCategorical ?? Resources.Load<ComputeShader>("GPU Kernels/SampleCategorical");
        static ComputeShader _sampleCategorical;

        public static ComputeShader setValues => _setValues = _setValues ?? Resources.Load<ComputeShader>("GPU Kernels/Set Value");
        static ComputeShader _setValues;

        public static ComputeShader transpose => _transpose = _transpose ?? Resources.Load<ComputeShader>("GPU Kernels/Transpose");
        static ComputeShader _transpose;

    }
}

