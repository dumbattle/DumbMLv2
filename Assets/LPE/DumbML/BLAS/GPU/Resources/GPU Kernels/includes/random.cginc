
float rand01(float2 seed) {
    return frac(sin(dot(seed, float2(12.9898, 78.233))) * 43758.5453);
}