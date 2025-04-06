using Unity.Jobs;
using Unity.Collections;
using System;
namespace DumbML.BLAS.CPU {
    public struct SampleCategoricalJob : IJobParallelFor {
        [ReadOnly]
        NativeArray<float> src;
        NativeArray<int> dest;

        int stride;
        int seed;
        int seedStep;

        public SampleCategoricalJob(FloatCPUTensorBuffer src, IntCPUTensorBuffer dest) {
            this.src = src.buffer;
            this.dest = dest.buffer;

            stride = src.shape[src.Rank() - 1];

            seed = UnityEngine.Random.Range(0, 1_000_000);
            seedStep = UnityEngine.Random.Range(100, 10_000);
        }

        public void Execute(int idx) {
            var rng = Unity.Mathematics.Random.CreateFromIndex((uint)(seed + idx * seedStep));
            float r = (float)rng.NextDouble();
            int result = 0;
            for (int i = 0; i < stride; i++) {
                r -= src[idx + i];

                if (r <= 0) {
                    result = i;
                    break;
                }
            }

            dest[idx] = result;
        }
    }
}
