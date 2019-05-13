extern "C"
{   


#define IDX_I	(idx(in_sx , out_depth, ix, iy,  d))
#define IDX_O	(idx(out_sx, out_depth, ox, oy,  d))

__device__ int idx (int sx, int depth, int x, int y, int d)
{
	return (y*sx + x)*depth + d;
}

__global__ void PoolForward (double* out_w, int* switch_xy,
							const double* in_w,
							int fi_sx , int fi_sy ,
							int in_sx , int in_sy ,
							int out_sx, int out_sy, int out_depth,
							int stride, int pad)
{
	int ox = blockDim.x * blockIdx.x + threadIdx.x; // (0 ~ out_sx)
	int oy = blockDim.y * blockIdx.y + threadIdx.y; // (0 ~ out_sy)
	int d  = blockDim.z * blockIdx.z + threadIdx.z; // (0 ~ out_depth == in_depth)

	int n = IDX_O;

	int x = -pad + ox*stride;
	int y = -pad + oy*stride;

	double	_a = -100000;
	int		winx = switch_xy[n*2 + 0];
	int		winy = switch_xy[n*2 + 1];

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

			double v = in_w[IDX_I];

			if (v > _a)
			{
				_a = v;
				winx = ix;
				winy = iy;
			}
		}
	}

	switch_xy[n*2 + 0] = winx;
	switch_xy[n*2 + 1] = winy;

	out_w[n] = _a;
}

__global__ void PoolBackward (double* in_dw,
							const double* out_dw, const int* switch_xy,
							int in_sx , int in_sy ,
							int out_sx, int out_sy, int out_depth)
{
	int ix = blockDim.x * blockIdx.x + threadIdx.x; // (0 ~ in_sx)
	int iy = blockDim.y * blockIdx.y + threadIdx.y; // (0 ~ in_sy)
	int d  = blockDim.z * blockIdx.z + threadIdx.z; // (0 ~ in_depth == out_depth)

	int idx_i = IDX_I;

	double sum_dw = 0;
	for (int oy = 0; oy < out_sy; oy++)
	{
		for (int ox = 0; ox < out_sx; ox++)
		{
			int n = IDX_O;
			int winx = switch_xy[n*2 + 0];
			int winy = switch_xy[n*2 + 1];

			if (winx == ix && winy == iy)
			{
				sum_dw += out_dw[n];
			}
		}
	}

	in_dw[idx_i] = sum_dw;
}

}
