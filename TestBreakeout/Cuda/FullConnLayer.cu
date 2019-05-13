extern "C"
{   

__global__ void FcForward (double* out_w, 
						const double* in_w, const double* fi_w,
						int inN, int fi_size)
{
	int v; int u; int m;
	v = blockDim.x * blockIdx.x + threadIdx.x; // (0 ~ out_depth)

	double output_w = 0;

	for (u = 0; u < inN; u++)
	{
		output_w += in_w[u] * fi_w[inN*v + u];
	}

	output_w += fi_w[fi_size + v];
		
	out_w[v] = output_w;
}
__global__ void FcBackward_u (double* in_dw, double* fi_dw,
						const double* in_w, const double* fi_w, const double* out_dw,
						int inN, int outN)
{
    int v; int u; int m;
	u = blockDim.x * blockIdx.x + threadIdx.x;	// (0 ~ num_inputs+1)

	double input_dw = 0;			// 새 값을 적용

	for (v = 0; v < outN; v++)
	{
		input_dw += fi_w[inN*v + u] * out_dw[v];
	}

	in_dw[u] = input_dw;
}
__global__ void FcBackward_v (double* in_dw, double* fi_dw,
						const double* in_w, const double* fi_w, const double* out_dw,
						int inN, int fi_size)
{
    int v; int u; int m;
	v = blockDim.x * blockIdx.x + threadIdx.x;	// (0 ~ out_depth)

	double c_g = out_dw[v]; // chain_grad

	for (u = 0; u < inN; u++)
	{
		fi_dw[inN*v + u] += in_w[u] * c_g;
	}

	fi_dw[fi_size + v] += c_g;	// 이전상태에서 누적
}

}
