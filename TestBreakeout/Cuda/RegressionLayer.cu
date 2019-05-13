extern "C"
{
// loss[0] : end layer loss

__global__ void RegressionLoss (double* loss, double* in_dw,
							const double* in_w, const  double* out_w)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

		double dy = in_w[i] - out_w[i];
		in_dw[i] = dy;
		double ladd = dy * dy / 2;
		atomicAdd (loss, ladd);
}

}
