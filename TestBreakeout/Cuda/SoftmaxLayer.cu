extern "C"
{   

__global__ void SoftmaxForw (double* out_w, double* es, const double* in_w, int N)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	const double* aw = in_w;

	// compute max activation
	__shared__ double amax;
	__shared__ double esum;
	if (i == 0)
	{
		amax = aw[0];
		esum = 0;
		for (int k = 1; k < N; k++)
			if (aw[k] > amax)
				amax = aw[k];
	}

	// compute exponentials (carefully to not blow up)
	__syncthreads();
		double e = exp (aw[i] - amax);
		atomicAdd (&esum, e);
		es[i] = e;

	// normalize and output to sum to one
	__syncthreads();
		es[i] /= esum;
		out_w[i] = es[i];
}
__global__ void SoftmaxLoss (double* loss, double* in_dw, double* es, int y, const double* out_w)
{
	int i = blockDim.x * blockIdx.x + threadIdx.x;

		double indicator = (y == i) ? 1 : 0;
		double mul = -(indicator - es[i]);
		in_dw[i] = mul;

	*loss = -log (es[y]);

	*loss = *loss < 1000 ? *loss : 1000;
}

}
