extern "C"
{   

__global__ void TanhForward (double* out_w, const double* in_w, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	out_w[i] = tanh(in_w[i]);
}
__global__ void TanhBackward (double* in_dw, const double* out_w, const double* out_dw, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	double w = out_w[i];
	in_dw[i] = (1.0 - w * w) * out_dw[i];
}

}
