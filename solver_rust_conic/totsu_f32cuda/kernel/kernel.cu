// nvcc --ptx kernel.cu

extern "C" __global__ void maximum(float* x, float v, int n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        x[i] = fmaxf(x[i], v);
    }
}

extern "C" __global__ void minimum(float* x, float v, int n)
{
    for(int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x)
    {
        x[i] = fminf(x[i], v);
    }
}
