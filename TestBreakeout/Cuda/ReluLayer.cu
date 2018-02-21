extern "C"
{   

__global__ void ReluForward (double* out_w, const double* in_w, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	out_w[i] = in_w[i] > 0 ? in_w[i] : 0;
}
__global__ void ReluBackward (double* in_dw, const double* out_w, const double* out_dw, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	in_dw[i] = out_w[i] > 0 ? out_dw[i] : 0;
}

}
