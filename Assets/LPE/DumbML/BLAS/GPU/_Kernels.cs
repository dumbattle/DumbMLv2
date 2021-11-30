using UnityEngine;

namespace DumbML.BLAS.GPU {
    public static class Kernels {
        public static ComputeShader elementWiseBinary => _elementWiseBinary = _elementWiseBinary ?? Resources.Load<ComputeShader>("GPU Kernels/Elementwise Binary");
        static ComputeShader _elementWiseBinary;

        public static ComputeShader elementWiseSingleParam => _elementWiseSingleParam = _elementWiseSingleParam ?? Resources.Load<ComputeShader>("GPU Kernels/Elementwise Single With Param");
        static ComputeShader _elementWiseSingleParam;

        public static ComputeShader elementWiseSingle => _elementWiseSingle = _elementWiseSingle ?? Resources.Load<ComputeShader>("GPU Kernels/Elementwise Single");
        static ComputeShader _elementWiseSingle;

        public static ComputeShader matrixMult => _matrixMult = _matrixMult ?? Resources.Load<ComputeShader>("GPU Kernels/Matrix Mult");
        static ComputeShader _matrixMult;

        public static ComputeShader setValues => _setValues = _setValues ?? Resources.Load<ComputeShader>("GPU Kernels/Set Value");
        static ComputeShader _setValues;

    }
}

