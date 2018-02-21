extern "C"
{   

__global__ void SigmoidForward (double* out_w, const double* in_w, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	out_w[i] = 1.0 / (1.0 + exp(-in_w[i]));
}
__global__ void SigmoidBackward (double* in_dw, const double* out_w, const double* out_dw, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	double w = out_w[i];
	in_dw[i] = w * (1 - w) * out_dw[i];
}

}
