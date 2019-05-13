extern "C"
{   

#define IDX_F	(idx(fi_sx , fi_depth , fx, fy, fd) + fi_pitch*d)
#define IDX_I	(idx(in_sx , in_depth , ix, iy, fd))
#define IDX_O	(idx(out_sx, out_depth, ox, oy,  d))

__device__ int idx (int sx, int depth, int x, int y, int d)
{
	return (y*sx + x)*depth + d;
}

__global__ void ConvForward (double* out_w,	
							const double* in_w, const double* fi_w,
							int fi_sx , int fi_sy , int fi_depth , int fi_pitch, int fi_size,
							int in_sx , int in_sy , int in_depth , 
							int out_sx, int out_sy, int out_depth,
							int stride, int pad)
{
	int ox = blockDim.x * blockIdx.x + threadIdx.x; // (0 ~ out_sx)
	int oy = blockDim.y * blockIdx.y + threadIdx.y; // (0 ~ out_sy)
	int d  = blockDim.z * blockIdx.z + threadIdx.z; // (0 ~ out_depth)

	int x = -pad + ox*stride;
	int y = -pad + oy*stride;

	double output_w = 0;

	for (int fy = 0; fy < fi_sy; fy++)
	{
		int iy = y + fy;
		if (iy < 0 || iy >= in_sy)
			continue;

		for (int fx = 0; fx < fi_sx; fx++)
		{
			int ix = x + fx;
			if (ix < 0 || ix >= in_sx)
				continue;

			for (int fd = 0; fd < fi_depth; fd++)
			{
				output_w += fi_w[IDX_F] * in_w[IDX_I];
			}
		}
	}

	output_w += fi_w[fi_size + d]; // biases

	out_w[IDX_O] = output_w;
}

__global__ void ConvBackward_In(double* in_dw,
							const double* fi_w, const double* out_dw,
							int fi_sx , int fi_sy   , int fi_depth , int fi_pitch,
							int in_sx , int in_depth,
							int out_sx, int out_sy  , int out_depth,
							int stride, int pad)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x; // (0 ~ in_sx)
	int iy = blockDim.y * blockIdx.y + threadIdx.y; // (0 ~ in_sy)
	int fd = blockDim.z * blockIdx.z + threadIdx.z; // (0 ~ in_depth = fi_depth)

	double input_dw = 0;					// 새 값을 적용

	int y = -pad;
	for (int oy = 0; oy < out_sy; oy++, y += stride)
	{
		int fy = iy - y;
		if (fy < 0 || fy >= fi_sy)
			continue;

		int x = -pad;
		for (int ox = 0; ox < out_sx; ox++, x += stride)
		{
			int fx = ix - x;
			if (fx < 0 || fx >= fi_sx)
				continue;

			for (int d = 0; d < out_depth; d++)
			{
				input_dw += fi_w[IDX_F] * out_dw[IDX_O];
			}
		}
	}

	in_dw[IDX_I] += input_dw;
}

__global__ void ConvBackward_Fi (double* fi_dw,
							const double* in_w, const double* out_dw,
							int fi_sx , int fi_depth, int fi_pitch ,
							int in_sx , int in_sy   , int in_depth , 
							int out_sx, int out_sy  , int out_depth,
							int stride, int pad)
{
	int fx = blockDim.x * blockIdx.x + threadIdx.x; // (0 ~ fi_sx)
	int fy = blockDim.y * blockIdx.y + threadIdx.y; // (0 ~ fi_sy)
	int fd = blockDim.z * blockIdx.z + threadIdx.z; // (0 ~ fi_depth = in_depth)

	for (int d = 0; d < out_depth; d++)
	{
		double filter_dw = 0;

		int y = -pad;
		for (int oy = 0; oy < out_sy; oy++, y += stride)
		{
			int iy = y + fy;
			if (iy < 0 || iy >= in_sy)
				continue;

			int x = -pad;
			for (int ox = 0; ox < out_sx; ox++, x += stride)
			{
				int ix = x + fx;
				if (ix < 0 || ix >= in_sx)
					continue;

				filter_dw += in_w[IDX_I] * out_dw[IDX_O];
			}
		}

		fi_dw[IDX_F] += filter_dw;			// 이전 상태에서 누적
	}
}

__global__ void ConvBackward_Bi (double* fi_dw, const double* out_dw, int fi_size,
							int out_sx, int out_sy, int out_depth, 
							int stride)
{
	int d = blockDim.z * blockIdx.z + threadIdx.z; // (0 ~ out_depth)

	double biases_dw = fi_dw[fi_size + d];

	for (int oy = 0; oy < out_sy; oy++)
	{
		for (int ox = 0; ox < out_sx; ox++)
		{
			biases_dw += out_dw[IDX_O];
		}
	}

	fi_dw[fi_size + d] += biases_dw;	// 이전 상태에서 누적
}

}
