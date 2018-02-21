extern "C"
{
__device__ int  __idx (int size1, int size0, int x2, int x1, int x0)
{
	return (x2*size1 + x1)*size0 + x0;
}
__device__ void __udx (int size1, int size0, int n, int* x2, int* x1, int* x0)
{
	int w;
	*x2 = n / (size1*size0);
	 w  = n % (size1*size0);
	*x1 = w / size0;
	*x0 = w % size0;
}

//
// input 
//
__device__ void NoneForw (int i, double* out_w, const double* in_w)
{
	out_w[i] = in_w[i];
}
__device__ void NoneBack (int i, double* in_dw, const double* out_dw)
{
	in_dw[i] = out_dw[i];
}
//
// Conv
//
#define IDX_F	(__idx(fi_sx , in_depth , fy, fx, fd) + fi_pitch*d)
#define IDX_I	(__idx(in_sx , in_depth , iy, ix, fd))
#define IDX_O	(__idx(out_sx, out_depth, oy, ox,  d))

__global__ void ConvForward (int ox, int oy, int d, double* out_w,
							const double* in_w, const double* fi_w,
							int in_sx , int in_sy , int in_depth , 
							int out_sx, int out_sy, int out_depth,
							int fi_sx , int fi_sy , int fi_pitch, int fi_size,
							int stride, int pad)
{
	//int ox = blockDim.x * blockIdx.x + threadIdx.x; // (0 ~ out_sx)
	//int oy = blockDim.y * blockIdx.y + threadIdx.y; // (0 ~ out_sy)
	//int d  = blockDim.z * blockIdx.z + threadIdx.z; // (0 ~ out_depth)

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

			for (int fd = 0; fd < in_depth; fd++)
			{
				output_w += fi_w[IDX_F] * in_w[IDX_I];
			}
		}
	}

	output_w += fi_w[fi_size + d];		// biases

	out_w[IDX_O] = output_w;
}
__global__ void ConvBackward_In(int ix, int iy, int fd, double* in_dw,
							const double* fi_w, const double* out_dw,
							int in_sx , int in_depth,
							int out_sx, int out_sy  , int out_depth,
							int fi_sx , int fi_sy   , int fi_pitch,
							int stride, int pad)
{
	//int ix = blockDim.x * blockIdx.x + threadIdx.x; // (0 ~ in_sx)
	//int iy = blockDim.y * blockIdx.y + threadIdx.y; // (0 ~ in_sy)
	//int fd = blockDim.z * blockIdx.z + threadIdx.z; // (0 ~ in_depth)

	double input_dw = 0;

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

	in_dw[IDX_I] = input_dw;			// 새 값을 적용
}
__global__ void ConvBackward_Fi (int fx, int fy, int fd, double* fi_dw,
							const double* in_w, const double* out_dw,
							int in_sx , int in_sy   , int in_depth , 
							int out_sx, int out_sy  , int out_depth,
							int fi_sx , int fi_pitch ,
							int stride, int pad)
{
	//int fx = blockDim.x * blockIdx.x + threadIdx.x; // (0 ~ fi_sx)
	//int fy = blockDim.y * blockIdx.y + threadIdx.y; // (0 ~ fi_sy)
	//int fd = blockDim.z * blockIdx.z + threadIdx.z; // (0 ~ in_depth)

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

		fi_dw[IDX_F] += filter_dw;		// 이전 상태에서 누적
	}
}
__global__ void ConvBackward_Bi (int d, double* fi_dw, const double* out_dw, int fi_size,
							int out_sx, int out_sy, int out_depth, 
							int stride)
{
	//int d = blockDim.z * blockIdx.z + threadIdx.z; // (0 ~ out_depth)

	double biases_dw = 0;

	for (int oy = 0; oy < out_sy; oy++)
	{
		for (int ox = 0; ox < out_sx; ox++)
		{
			biases_dw += out_dw[IDX_O];
		}
	}

	fi_dw[fi_size + d] += biases_dw;	// 이전 상태에서 누적
}
//
// Pool
//
#undef  IDX_I
#define IDX_I	(__idx(in_sx , out_depth, iy, ix, d))
#define IDX_O	(__idx(out_sx, out_depth, oy, ox, d))

__global__ void PoolForward (int n, int ox, int oy, int d, double* out_w, int* switch_x, int* switch_y,
							const double* in_w,
							int in_sx , int in_sy ,
							int out_sx, int out_sy, int out_depth,
							int fi_sx , int fi_sy ,
							int stride, int pad)
{
	//int ox = blockDim.x * blockIdx.x + threadIdx.x; // (0 ~ out_sx)
	//int oy = blockDim.y * blockIdx.y + threadIdx.y; // (0 ~ out_sy)
	//int d  = blockDim.z * blockIdx.z + threadIdx.z; // (0 ~ out_depth == in_depth)

	int x = -pad + ox*stride;
	int y = -pad + oy*stride;

	double	a = -100000;
	int		winx = switch_x[n];
	int		winy = switch_y[n];

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
			if (v > a) { a = v; winx = ix; winy = iy; }
		}
	}

	switch_x[n] = winx;
	switch_y[n] = winy;
	int out_idx = __idx(out_sx, out_depth, oy, ox, d);
	out_w[out_idx] = a;
}
__global__ void PoolBackward (int n, int ox, int oy, int d, double* in_dw,
							const double* out_dw, const int* switch_x, const int* switch_y,
							int in_sx , int in_sy ,
							int out_sx, int out_sy, int out_depth)
{
	int winx = switch_x[n];
	int winy = switch_y[n];
	int Iidx = __idx(in_sx , out_depth, winy, winx, d);
	int Oidx = __idx(out_sx, out_depth,   oy,   ox, d);

	//atomicAdd (in_dw + Iidx, out_dw[Oidx]);
	in_dw[Iidx] += out_dw[Oidx];
}
__global__ void PoolBackward__ (int n, int ix, int iy, int d, double* in_dw,
							const double* out_dw, const int* switch_x, const int* switch_y,
							int in_sx , int in_sy ,
							int out_sx, int out_sy, int out_depth)
{
	//int ix = blockDim.x * blockIdx.x + threadIdx.x; // (0 ~ in_sx)
	//int iy = blockDim.y * blockIdx.y + threadIdx.y; // (0 ~ in_sy)
	//int d  = blockDim.z * blockIdx.z + threadIdx.z; // (0 ~ in_depth == out_depth)

	int idx_i = IDX_I;

	double sum_dw = 0;
	for (int oy = 0; oy < out_sy; oy++)
	{
		for (int ox = 0; ox < out_sx; ox++)
		{
			int n = IDX_O;
			int winx = switch_x[n];
			int winy = switch_y[n];

			if (winx == ix && winy == iy)
			{
				sum_dw += out_dw[n];
			}
		}
	}

	in_dw[idx_i] = sum_dw;
}
//
// Fc
//
__device__ void FcForward    (int v, double* out_w, 
							const double* in_w, const double* fi_w,
							int inN, int fi_size)
{
	int u; double out_w__ = 0;

	for (u = 0; u < inN; u++)
	{
		out_w__ += in_w[u] * fi_w[inN*v + u];
	}

	out_w__ += fi_w[fi_size + v];
		
	out_w[v] = out_w__;
}
__device__ void FcBackward_u (int u, double* in_dw, double* fi_dw,
							const double* in_w, const double* fi_w, const double* out_dw,
							int inN, int outN)
{
	int v; double in_dw__ = 0;			// 새 값을 적용

	for (v = 0; v < outN; v++)
	{
		in_dw__ += fi_w[inN*v + u] * out_dw[v];
	}

	in_dw[u] = in_dw__;
}
__device__ void FcBackward_v (int v, double* in_dw, double* fi_dw,
							const double* in_w, const double* fi_w, const double* out_dw,
							int inN, int fi_size)
{
	int u; double c_g = out_dw[v];		// chain_grad

	for (u = 0; u < inN; u++)
	{
		fi_dw[inN*v + u] += in_w[u] * c_g;
	}

	fi_dw[fi_size + v] += c_g;			// 이전상태에서 누적
}
//
// Relu
//
__device__ void ReluForward     (int i, double* out_w, const double* in_w)
{
	out_w[i] = in_w[i] > 0 ? in_w[i] : 0;
}
__device__ void ReluBackward    (int i, double* in_dw, const double* out_w, const double* out_dw)
{
	in_dw[i] = out_w[i] > 0 ? out_dw[i] : 0;
}
//
// Sigmoid
//
__device__ void SigmoidForward  (int i, double* out_w, const double* in_w)
{
	out_w[i] = 1.0 / (1.0 + exp(-in_w[i]));
}
__device__ void SigmoidBackward (int i, double* in_dw, const double* out_w, const double* out_dw)
{
	double w = out_w[i];
	in_dw[i] = w * (1 - w) * out_dw[i];
}
//
// Regression
//
__device__ void RegressionLoss (int i, double* loss, double* in_dw, const double* in_w, const  double* out_w)
{
	double dy = in_w[i] - out_w[i];
	in_dw[i] = dy;
	double ladd = dy * dy / 2;

	//*loss += ladd;
	atomicAdd (loss, ladd);
}
//
// Softmax
//
__global__ void SoftmaxForward (int i, double* out_w, double* es, const double* in_w, int N)
{
	const double* aw = in_w;

	// compute max activation
	__shared__ double amax;
	if (i == 0)
	{
		amax = in_w[0];
		for (int k = 0; k < N; k++)
			if (aw[k] > amax)
				amax = aw[k];
	}

	// compute exponentials (carefully to not blow up)
	__shared__ double esum;
	__syncthreads();
		double e = exp (aw[i] - amax);
		//esum += e;
		atomicAdd (&esum, e);
		es[i] = e;

	// normalize and output to sum to one
	__syncthreads();
		es[i] /= esum;
		out_w[i] = es[i];
}
__global__ void SoftmaxForward_m (double* amax, const double* aw, int N)
{
	// compute max activation
		*amax = aw[0];
		for (int k = 1; k < N; k++)
			if (aw[k] > *amax)
				*amax = aw[k];
}
__global__ void SoftmaxForward_s (int i, double* esum, double* es, double amax, const double* aw)
{
	// compute exponentials (carefully to not blow up)
		double e = exp (aw[i] - amax);
		//*esum += e;
		atomicAdd (esum, e);
		es[i] = e;
}
__global__ void SoftmaxForward_n (int i, double esum, double* es, double* out_w)
{
	// normalize and output to sum to one
		es[i] /= esum;
		out_w[i] = es[i];
}
__global__ void SoftmaxLoss (int i, double* loss, double* in_dw, double* es, const double* out_w)
{
		double indicator = out_w[i];//(y == i) ? 1 : 0;
		double mul = -(indicator - es[i]);
		in_dw[i] = mul;

	if (out_w[i] == 1)
	{
		double ls = -log (es[i]);
		*loss += ls < 1000 ? ls : 1000;
	}
}
//
// Train
//
__device__ void sgd_nm (int j, double* loss, double* p, double* g, 
						double decay_2, double decay_1, int batch_size, double learning_rate)
{
	double p_j = p[j];

	double l2add = decay_2 * p_j * p_j / 2;		// loss_2
	double l1add = decay_1 * abs(p_j);			// loss_1
	atomicAdd (loss + 0, l2add);
	atomicAdd (loss + 1, l1add);

	double l2grad = decay_2 * (p_j);
	double l1grad = decay_1 * (p_j > 0 ? 1 : -1);

	double gij = (l2grad + l1grad + g[j]) / batch_size;

	atomicAdd (p + j, -learning_rate * gij);	//p[j] += -learning_rate * gij;

	g[j] = 0;
}
__global__ void sgd_mome (int j, double* loss, double* p, double* g, double* gsumi,
						double l2_decay, double l1_decay, int batch_size, double learning_rate, double momentum)
{
	double p_j = p[j];

	double l2add = l2_decay * p[j] * p_j / 2;	// loss_2
	double l1add = l1_decay * abs(p_j);			// loss_1
	atomicAdd (loss + 0, l2add);
	atomicAdd (loss + 1, l1add);

	double l2grad = l2_decay * (p_j);
	double l1grad = l1_decay * (p_j > 0 ? 1 : -1);

	double gij = (l2grad + l1grad + g[j]) / batch_size;

	double dx = momentum * gsumi[j] - learning_rate * gij;
	atomicAdd (gsumi + j, dx);					// gsumi[j] = dx;
	atomicAdd (p + j, dx);						// p[j] += dx;

	g[j] = 0;
}

__device__ int div_ceil (int a, int b) { return (a+b-1)/b; }

#define VInS    		3072
#define VOutS   		10

#define OutS_0  		3072
#define OutS_1  		16384
#define OutS_2  		16384
#define OutS_3  		4096
#define OutS_4  		5120
#define OutS_5  		5120
#define OutS_6  		1280
#define OutS_7  		1280
#define OutS_8  		1280
#define OutS_9  		320
#define OutS_10  		10
#define OutS_11  		10

#define vo0_w   		(dbufs + 0)
#define vo0_dw  		(dbufs + 3072)
#define vo1_w   		(dbufs + 6144)
#define vo1_dw  		(dbufs + 22528)
#define vo2_w   		(dbufs + 38912)
#define vo2_dw  		(dbufs + 55296)
#define vo3_w   		(dbufs + 71680)
#define vo3_dw  		(dbufs + 75776)
#define vo4_w   		(dbufs + 79872)
#define vo4_dw  		(dbufs + 84992)
#define vo5_w   		(dbufs + 90112)
#define vo5_dw  		(dbufs + 95232)
#define vo6_w   		(dbufs + 100352)
#define vo6_dw  		(dbufs + 101632)
#define vo7_w   		(dbufs + 102912)
#define vo7_dw  		(dbufs + 104192)
#define vo8_w   		(dbufs + 105472)
#define vo8_dw  		(dbufs + 106752)
#define vo9_w   		(dbufs + 108032)
#define vo9_dw  		(dbufs + 108352)
#define vo10_w   		(dbufs + 108672)
#define vo10_dw  		(dbufs + 108682)
#define vo11_w   		(dbufs + 108692)
#define vo11_dw  		(dbufs + 108702)
#define fl1_dw  		(dbufs + 108712)
#define fl4_dw  		(dbufs + 109928)
#define fl7_dw  		(dbufs + 117948)
#define fl10_dw  		(dbufs + 127968)
#define switchX_3   	(ibufs + 0)
#define switchY_3   	(ibufs + 4096)
#define switchX_6   	(ibufs + 8192)
#define switchY_6   	(ibufs + 9472)
#define switchX_9   	(ibufs + 10752)
#define switchY_9   	(ibufs + 11072)

#define OutDepth_0  	3
#define OutSx_0     	32
#define OutSy_0     	32

#define CvFiSx_1    	5
#define CvFiSy_1    	5
#define CvStride_1  	1
#define CvPad_1     	2
#define FiPitch_1   	CvFiSx_1*CvFiSy_1*OutDepth_0
#define FiSize_1    	FiPitch_1*OutDepth_1
#define FiBiS_1     	FiPitch_1*OutDepth_1+OutDepth_1
#define FiBiP_1     	0
#define DecayP_1    	0
#define OutDepth_1  	16
#define OutSx_1     	32
#define OutSy_1     	32

#define OutDepth_2  	16
#define OutSx_2     	32
#define OutSy_2     	32

#define PoFiSx_3    	2
#define PoFiSy_3    	2
#define PoStride_3  	2
#define PoPad_3     	0
#define OutDepth_3  	16
#define OutSx_3     	16
#define OutSy_3     	16

#define CvFiSx_4    	5
#define CvFiSy_4    	5
#define CvStride_4  	1
#define CvPad_4     	2
#define FiPitch_4   	CvFiSx_4*CvFiSy_4*OutDepth_3
#define FiSize_4    	FiPitch_4*OutDepth_4
#define FiBiS_4     	FiPitch_4*OutDepth_4+OutDepth_4
#define FiBiP_4     	FiBiP_1+FiBiS_1
#define DecayP_4    	4
#define OutDepth_4  	20
#define OutSx_4     	16
#define OutSy_4     	16

#define OutDepth_5  	20
#define OutSx_5     	16
#define OutSy_5     	16

#define PoFiSx_6    	2
#define PoFiSy_6    	2
#define PoStride_6  	2
#define PoPad_6     	0
#define OutDepth_6  	20
#define OutSx_6     	8
#define OutSy_6     	8

#define CvFiSx_7    	5
#define CvFiSy_7    	5
#define CvStride_7  	1
#define CvPad_7     	2
#define FiPitch_7   	CvFiSx_7*CvFiSy_7*OutDepth_6
#define FiSize_7    	FiPitch_7*OutDepth_7
#define FiBiS_7     	FiPitch_7*OutDepth_7+OutDepth_7
#define FiBiP_7     	FiBiP_4+FiBiS_4
#define DecayP_7    	8
#define OutDepth_7  	20
#define OutSx_7     	8
#define OutSy_7     	8

#define OutDepth_8  	20
#define OutSx_8     	8
#define OutSy_8     	8

#define PoFiSx_9    	2
#define PoFiSy_9    	2
#define PoStride_9  	2
#define PoPad_9     	0
#define OutDepth_9  	20
#define OutSx_9     	4
#define OutSy_9     	4

#define FiSize_10    	OutS_10*OutS_9
#define FiBiS_10     	OutS_10*(OutS_9+1)
#define FiBiP_10     	FiBiP_7+FiBiS_7
#define DecayP_10    	12
#define OutDepth_10  	10
#define OutSx_10     	1
#define OutSy_10     	1

#define OutDepth_11  	10
#define OutSx_11     	1
#define OutSy_11     	1

__global__ void TestVolume (const double* in_w, double* out_w, 
							const double* fb_w, double* dbufs, int* ibufs)
{
	int maxit = blockDim.x;
	int it = threadIdx.x;
	int rep;

	// 0 --> Input
	rep = div_ceil (OutS_0 - it, maxit);
	for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		NoneForw (ii, vo0_w, in_w);

	// 1 --> Conv
	__syncthreads();
	rep = div_ceil (OutSx_1*OutSy_1*OutDepth_1 - it, maxit);
	for (int c = 0, ii = it; c < rep; c++, ii += maxit)
	{
		int x, y, z; __udx (OutSx_1, OutDepth_1, ii, &y, &x, &z);

		ConvForward (x, y, z, vo1_w, vo0_w, fb_w + FiBiP_1,
			OutSx_0 , OutSy_0 , OutDepth_0,
			OutSx_1 , OutSy_1 , OutDepth_1,
			CvFiSx_1, CvFiSy_1, FiPitch_1, FiPitch_1*OutDepth_1,
			CvStride_1, CvPad_1);
	}

	// 2 --> Relu
	rep = div_ceil (OutS_2 - it, maxit);
	for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		ReluForward (ii, vo2_w, vo1_w);

	// 3 --> Pool
	__syncthreads();
	rep = div_ceil (OutSx_3*OutSy_3*OutDepth_3 - it, maxit);
	for (int c = 0, ii = it; c < rep; c++, ii += maxit)
	{
		int x, y, d; __udx (OutSx_3, OutSy_3, ii, &d, &x, &y);

		PoolForward (ii, x, y, d, vo3_w, switchX_3, switchY_3, vo2_w,
			OutSx_1   , OutSy_1   ,
			OutSx_3   , OutSy_3   , OutDepth_3,
			PoFiSx_3  , PoFiSy_3  , 
			PoStride_3, PoPad_3);
	}

	// 4 --> Conv
	__syncthreads();
	rep = div_ceil (OutSx_4*OutSy_4*OutDepth_4 - it, maxit);
	for (int c = 0, ii = it; c < rep; c++, ii += maxit)
	{
		int x, y, z; __udx (OutSx_4, OutDepth_4, ii, &y, &x, &z);

		ConvForward (x, y, z, vo4_w, vo3_w, fb_w + FiBiP_4,
			OutSx_3 , OutSy_3 , OutDepth_3,
			OutSx_4 , OutSy_4 , OutDepth_4,
			CvFiSx_4, CvFiSy_4, FiPitch_4, FiPitch_4*OutDepth_4,
			CvStride_4, CvPad_4);
	}

	// 5 --> Relu
	rep = div_ceil (OutS_5 - it, maxit);
	for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		ReluForward (ii, vo5_w, vo4_w);

	// 6 --> Pool
	__syncthreads();
	rep = div_ceil (OutSx_6*OutSy_6*OutDepth_6 - it, maxit);
	for (int c = 0, ii = it; c < rep; c++, ii += maxit)
	{
		int x, y, d; __udx (OutSx_6, OutSy_6, ii, &d, &x, &y);

		PoolForward (ii, x, y, d, vo6_w, switchX_6, switchY_6, vo5_w,
			OutSx_4   , OutSy_4   ,
			OutSx_6   , OutSy_6   , OutDepth_6,
			PoFiSx_6  , PoFiSy_6  , 
			PoStride_6, PoPad_6);
	}

	// 7 --> Conv
	__syncthreads();
	rep = div_ceil (OutSx_7*OutSy_7*OutDepth_7 - it, maxit);
	for (int c = 0, ii = it; c < rep; c++, ii += maxit)
	{
		int x, y, z; __udx (OutSx_7, OutDepth_7, ii, &y, &x, &z);

		ConvForward (x, y, z, vo7_w, vo6_w, fb_w + FiBiP_7,
			OutSx_6 , OutSy_6 , OutDepth_6,
			OutSx_7 , OutSy_7 , OutDepth_7,
			CvFiSx_7, CvFiSy_7, FiPitch_7, FiPitch_7*OutDepth_7,
			CvStride_7, CvPad_7);
	}

	// 8 --> Relu
	rep = div_ceil (OutS_8 - it, maxit);
	for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		ReluForward (ii, vo8_w, vo7_w);

	// 9 --> Pool
	__syncthreads();
	rep = div_ceil (OutSx_9*OutSy_9*OutDepth_9 - it, maxit);
	for (int c = 0, ii = it; c < rep; c++, ii += maxit)
	{
		int x, y, d; __udx (OutSx_9, OutSy_9, ii, &d, &x, &y);

		PoolForward (ii, x, y, d, vo9_w, switchX_9, switchY_9, vo8_w,
			OutSx_7   , OutSy_7   ,
			OutSx_9   , OutSy_9   , OutDepth_9,
			PoFiSx_9  , PoFiSy_9  , 
			PoStride_9, PoPad_9);
	}

	// 10 --> Fc
	__syncthreads();
	rep = div_ceil (OutS_10 - it, maxit);
	for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		FcForward (ii, vo10_w, vo9_w, fb_w + FiBiP_10, OutS_9, OutS_9*OutS_10);

	// 11 --> Softmax
	__shared__ double amax;
	__shared__ double esum;
	double* aw = vo10_w;
	double* es = vo11_w;
	{
		amax = 0; esum = 0;
		rep = div_ceil (OutS_11 - it, maxit);
		__syncthreads();
		if (it == 0)
			SoftmaxForward_m (&amax, aw, OutS_11);
		__syncthreads();
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			SoftmaxForward_s (ii, &esum, es, amax, aw);
		__syncthreads();
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			SoftmaxForward_n (ii, esum, es, vo11_w);
	}
}
__global__ void TrainEpoch (int sample_num, int epoch, int start_pos, 
							const double* in_w, const double* out_w,
							double* fb_w, double* dbufs, int* ibufs, double* __fdecay, 
							double* __loss, double* __gsumi, double* __xsumi,
							double l2_decay, double l1_decay, int batch_size, double learning_rate, double momentum)
{
	int maxit = blockDim.x;
	int it = threadIdx.x;
	int rep;
	__loss[0] = 0;
	__loss[1] = 0;
	__loss[2] = 0;

	for (int c = 0; c < epoch; c++)
	{
		int s = (c + start_pos) % sample_num;

		// 0 --> Input
		rep = div_ceil (OutS_0 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			NoneForw (ii, vo0_w, in_w + s * VInS);

		// 1 --> Conv
		__syncthreads();
		rep = div_ceil (OutSx_1*OutSy_1*OutDepth_1 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			int x, y, z; __udx (OutSx_1, OutDepth_1, ii, &y, &x, &z);

			ConvForward (x, y, z, vo1_w, vo0_w, fb_w + FiBiP_1,
					OutSx_0 , OutSy_0 , OutDepth_0,
					OutSx_1 , OutSy_1 , OutDepth_1,
					CvFiSx_1, CvFiSy_1, FiPitch_1, FiPitch_1*OutDepth_1,
					CvStride_1, CvPad_1);
		}

		// 2 --> Relu
		rep = div_ceil (OutS_2 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			ReluForward (ii, vo2_w, vo1_w);

		// 3 --> Pool
		__syncthreads();
		rep = div_ceil (OutSx_3*OutSy_3*OutDepth_3 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			int x, y, d; __udx (OutSx_3, OutSy_3, ii, &d, &x, &y);

			PoolForward (ii, x, y, d, vo3_w, switchX_3, switchY_3, vo2_w,
					OutSx_1   , OutSy_1   ,
					OutSx_3   , OutSy_3   , OutDepth_3,
					PoFiSx_3  , PoFiSy_3  , 
					PoStride_3, PoPad_3);
		}

		// 4 --> Conv
		__syncthreads();
		rep = div_ceil (OutSx_4*OutSy_4*OutDepth_4 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			int x, y, z; __udx (OutSx_4, OutDepth_4, ii, &y, &x, &z);

			ConvForward (x, y, z, vo4_w, vo3_w, fb_w + FiBiP_4,
					OutSx_3 , OutSy_3 , OutDepth_3,
					OutSx_4 , OutSy_4 , OutDepth_4,
					CvFiSx_4, CvFiSy_4, FiPitch_4, FiPitch_4*OutDepth_4,
					CvStride_4, CvPad_4);
		}

		// 5 --> Relu
		rep = div_ceil (OutS_5 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			ReluForward (ii, vo5_w, vo4_w);

		// 6 --> Pool
		__syncthreads();
		rep = div_ceil (OutSx_6*OutSy_6*OutDepth_6 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			int x, y, d; __udx (OutSx_6, OutSy_6, ii, &d, &x, &y);

			PoolForward (ii, x, y, d, vo6_w, switchX_6, switchY_6, vo5_w,
					OutSx_4   , OutSy_4   ,
					OutSx_6   , OutSy_6   , OutDepth_6,
					PoFiSx_6  , PoFiSy_6  , 
					PoStride_6, PoPad_6);
		}

		// 7 --> Conv
		__syncthreads();
		rep = div_ceil (OutSx_7*OutSy_7*OutDepth_7 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			int x, y, z; __udx (OutSx_7, OutDepth_7, ii, &y, &x, &z);

			ConvForward (x, y, z, vo7_w, vo6_w, fb_w + FiBiP_7,
					OutSx_6 , OutSy_6 , OutDepth_6,
					OutSx_7 , OutSy_7 , OutDepth_7,
					CvFiSx_7, CvFiSy_7, FiPitch_7, FiPitch_7*OutDepth_7,
					CvStride_7, CvPad_7);
		}

		// 8 --> Relu
		rep = div_ceil (OutS_8 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			ReluForward (ii, vo8_w, vo7_w);

		// 9 --> Pool
		__syncthreads();
		rep = div_ceil (OutSx_9*OutSy_9*OutDepth_9 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			int x, y, d; __udx (OutSx_9, OutSy_9, ii, &d, &x, &y);

			PoolForward (ii, x, y, d, vo9_w, switchX_9, switchY_9, vo8_w,
					OutSx_7   , OutSy_7   ,
					OutSx_9   , OutSy_9   , OutDepth_9,
					PoFiSx_9  , PoFiSy_9  , 
					PoStride_9, PoPad_9);
		}

		// 10 --> Fc
		__syncthreads();
		rep = div_ceil (OutS_10 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			FcForward (ii, vo10_w, vo9_w, fb_w + FiBiP_10, OutS_9, OutS_9*OutS_10);

		// 11 --> Softmax
		__shared__ double amax;
		__shared__ double esum;
		double* aw = vo10_w;
		double* es = vo11_w;
		{
			amax = 0; esum = 0;
			rep = div_ceil (OutS_11 - it, maxit);
			__syncthreads();
			if (it == 0)
					SoftmaxForward_m (&amax, aw, OutS_11);
			__syncthreads();
			for (int c = 0, ii = it; c < rep; c++, ii += maxit)
					SoftmaxForward_s (ii, &esum, es, amax, aw);
			__syncthreads();
			for (int c = 0, ii = it; c < rep; c++, ii += maxit)
					SoftmaxForward_n (ii, esum, es, vo11_w);
		}

		// 11 <-- Softmax --------------------------------------------------------------------
		__syncthreads();
		rep = div_ceil (OutS_11 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			SoftmaxLoss (ii, __loss, vo10_dw, es, out_w + s * VOutS);

		// 10 <-- Fc
		__syncthreads();
		rep = div_ceil (OutS_9 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			FcBackward_u (ii, vo9_dw, fl10_dw, vo9_w, fb_w + FiBiP_10, vo10_dw, OutS_9, OutS_10);
		rep = div_ceil (OutS_10 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			FcBackward_v (ii, vo9_dw, fl10_dw, vo9_w, fb_w + FiBiP_10, vo10_dw, OutS_9, OutS_9*OutS_10);

		// 9 <-- Pool
		__syncthreads();
		rep = div_ceil (OutSx_7*OutSy_7*OutDepth_9 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			vo8_dw[ii] = 0;
		}
		__syncthreads();
		rep = div_ceil (OutSx_9*OutSy_9*OutDepth_9 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			int x, y, d; __udx (OutSx_9, OutSy_9, ii, &d, &x, &y);

			PoolBackward (ii, x, y, d, vo8_dw,
					vo9_dw    , switchX_9 , switchY_9 ,
					OutSx_7   , OutSy_7   ,
					OutSx_9   , OutSy_9   , OutDepth_9);
		}
		__syncthreads();

		// 8 <-- Relu
		rep = div_ceil (OutS_8 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			ReluBackward (ii, vo7_dw, vo8_w, vo8_dw);

		// 7 <-- Conv
		__syncthreads();
		rep = div_ceil (OutSx_6*OutSy_6*OutDepth_6 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			int x, y, z; __udx (OutSx_6, OutDepth_6, ii, &y, &x, &z);

			ConvBackward_In (x, y, z, vo6_dw,
					fb_w + FiBiP_7, vo7_dw,
					OutSx_6   , OutDepth_6,
					OutSx_7   , OutSy_7   , OutDepth_7,
					CvFiSx_7  , CvFiSy_7  , FiPitch_7,
					CvStride_7, CvPad_7);
		}
		__syncthreads();
		rep = div_ceil (CvFiSx_7*CvFiSy_7*OutDepth_6 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			int x, y, z; __udx (CvFiSx_7, OutDepth_6, ii, &y, &x, &z);

			ConvBackward_Fi (x, y, z, fl7_dw, 
					vo6_w, vo7_dw,
					OutSx_6   , OutSy_6   , OutDepth_6,
					OutSx_7   , OutSy_7   , OutDepth_7,
					CvFiSx_7  , FiPitch_7 ,
					CvStride_7, CvPad_7);
		}
		__syncthreads();
		rep = div_ceil (OutDepth_7 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			ConvBackward_Bi (ii, fl7_dw, vo7_dw, FiPitch_7*OutDepth_7,
					OutSx_7   , OutSy_7   , OutDepth_7,
					CvStride_7);

		// 6 <-- Pool
		__syncthreads();
		rep = div_ceil (OutSx_4*OutSy_4*OutDepth_6 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			vo5_dw[ii] = 0;
		}
		__syncthreads();
		rep = div_ceil (OutSx_6*OutSy_6*OutDepth_6 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			int x, y, d; __udx (OutSx_6, OutSy_6, ii, &d, &x, &y);

			PoolBackward (ii, x, y, d, vo5_dw,
					vo6_dw    , switchX_6 , switchY_6 ,
					OutSx_4   , OutSy_4   ,
					OutSx_6   , OutSy_6   , OutDepth_6);
		}
		__syncthreads();

		// 5 <-- Relu
		rep = div_ceil (OutS_5 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			ReluBackward (ii, vo4_dw, vo5_w, vo5_dw);

		// 4 <-- Conv
		__syncthreads();
		rep = div_ceil (OutSx_3*OutSy_3*OutDepth_3 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			int x, y, z; __udx (OutSx_3, OutDepth_3, ii, &y, &x, &z);

			ConvBackward_In (x, y, z, vo3_dw,
					fb_w + FiBiP_4, vo4_dw,
					OutSx_3   , OutDepth_3,
					OutSx_4   , OutSy_4   , OutDepth_4,
					CvFiSx_4  , CvFiSy_4  , FiPitch_4,
					CvStride_4, CvPad_4);
		}
		__syncthreads();
		rep = div_ceil (CvFiSx_4*CvFiSy_4*OutDepth_3 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			int x, y, z; __udx (CvFiSx_4, OutDepth_3, ii, &y, &x, &z);

			ConvBackward_Fi (x, y, z, fl4_dw, 
					vo3_w, vo4_dw,
					OutSx_3   , OutSy_3   , OutDepth_3,
					OutSx_4   , OutSy_4   , OutDepth_4,
					CvFiSx_4  , FiPitch_4 ,
					CvStride_4, CvPad_4);
		}
		__syncthreads();
		rep = div_ceil (OutDepth_4 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			ConvBackward_Bi (ii, fl4_dw, vo4_dw, FiPitch_4*OutDepth_4,
					OutSx_4   , OutSy_4   , OutDepth_4,
					CvStride_4);

		// 3 <-- Pool
		__syncthreads();
		rep = div_ceil (OutSx_1*OutSy_1*OutDepth_3 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			vo2_dw[ii] = 0;
		}
		__syncthreads();
		rep = div_ceil (OutSx_3*OutSy_3*OutDepth_3 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			int x, y, d; __udx (OutSx_3, OutSy_3, ii, &d, &x, &y);

			PoolBackward (ii, x, y, d, vo2_dw,
					vo3_dw    , switchX_3 , switchY_3 ,
					OutSx_1   , OutSy_1   ,
					OutSx_3   , OutSy_3   , OutDepth_3);
		}
		__syncthreads();

		// 2 <-- Relu
		rep = div_ceil (OutS_2 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			ReluBackward (ii, vo1_dw, vo2_w, vo2_dw);

		// 1 <-- Conv
		__syncthreads();
		rep = div_ceil (OutSx_0*OutSy_0*OutDepth_0 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			int x, y, z; __udx (OutSx_0, OutDepth_0, ii, &y, &x, &z);

			ConvBackward_In (x, y, z, vo0_dw,
					fb_w + FiBiP_1, vo1_dw,
					OutSx_0   , OutDepth_0,
					OutSx_1   , OutSy_1   , OutDepth_1,
					CvFiSx_1  , CvFiSy_1  , FiPitch_1,
					CvStride_1, CvPad_1);
		}
		__syncthreads();
		rep = div_ceil (CvFiSx_1*CvFiSy_1*OutDepth_0 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
		{
			int x, y, z; __udx (CvFiSx_1, OutDepth_0, ii, &y, &x, &z);

			ConvBackward_Fi (x, y, z, fl1_dw, 
					vo0_w, vo1_dw,
					OutSx_0   , OutSy_0   , OutDepth_0,
					OutSx_1   , OutSy_1   , OutDepth_1,
					CvFiSx_1  , FiPitch_1 ,
					CvStride_1, CvPad_1);
		}
		__syncthreads();
		rep = div_ceil (OutDepth_1 - it, maxit);
		for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			ConvBackward_Bi (ii, fl1_dw, vo1_dw, FiPitch_1*OutDepth_1,
					OutSx_1   , OutSy_1   , OutDepth_1,
					CvStride_1);


		// filter dw update -----------------------------------------------------------------
		if ((c+1) % batch_size == 0)
		{
			__syncthreads();
			rep = div_ceil (FiBiS_1 - it, maxit);
			for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			{
				int     j = ii;
				int     dp = (ii < FiSize_1) ? 0 : 2;
				double* train_loss = __loss + 1;
				double* p       = fb_w + FiBiP_1;
				double* g       = fl1_dw;
				double  decay_2 = l2_decay * __fdecay[DecayP_1 + dp    ];
				double  decay_1 = l1_decay * __fdecay[DecayP_1 + dp + 1];
				double* gsumi   = __gsumi + FiBiP_1;
				double* xsumi   = __xsumi + FiBiP_1;

				double p_j = p[j];

				double l2add = decay_2 * p_j * p_j / 2;		// loss_2
				double l1add = decay_1 * abs(p_j);			// loss_1
				train_loss[0] += l2add;
				train_loss[1] += l1add;
				//atomicAdd (train_loss + 0, l2add);
				//atomicAdd (train_loss + 1, l1add);

				double l2grad = decay_2 * (p_j);
				double l1grad = decay_1 * (p_j > 0 ? 1 : -1);

				double gij = (l2grad + l1grad + g[j]) / batch_size;

				if (momentum > 0)
				{
					double dx = momentum * gsumi[j] - learning_rate * gij;
					gsumi[j] = dx;
					p[j] += dx;
					//  atomicAdd (p + j, dx);						// p[j] += dx;
				}
				else
				{
					p[j] += -learning_rate * gij;
					//  atomicAdd (p + j, -learning_rate * gij);	// p[j] += -learning_rate * gij;
				}

				g[j] = 0;
			}

			__syncthreads();
			rep = div_ceil (FiBiS_4 - it, maxit);
			for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			{
				int     j = ii;
				int     dp = (ii < FiSize_4) ? 0 : 2;
				double* train_loss = __loss + 1;
				double* p       = fb_w + FiBiP_4;
				double* g       = fl4_dw;
				double  decay_2 = l2_decay * __fdecay[DecayP_4 + dp    ];
				double  decay_1 = l1_decay * __fdecay[DecayP_4 + dp + 1];
				double* gsumi   = __gsumi + FiBiP_4;
				double* xsumi   = __xsumi + FiBiP_4;

				double p_j = p[j];

				double l2add = decay_2 * p_j * p_j / 2;		// loss_2
				double l1add = decay_1 * abs(p_j);			// loss_1
				train_loss[0] += l2add;
				train_loss[1] += l1add;
				//atomicAdd (train_loss + 0, l2add);
				//atomicAdd (train_loss + 1, l1add);

				double l2grad = decay_2 * (p_j);
				double l1grad = decay_1 * (p_j > 0 ? 1 : -1);

				double gij = (l2grad + l1grad + g[j]) / batch_size;

				if (momentum > 0)
				{
					double dx = momentum * gsumi[j] - learning_rate * gij;
					gsumi[j] = dx;
					p[j] += dx;
					//  atomicAdd (p + j, dx);						// p[j] += dx;
				}
				else
				{
					p[j] += -learning_rate * gij;
					//  atomicAdd (p + j, -learning_rate * gij);	// p[j] += -learning_rate * gij;
				}

				g[j] = 0;
			}

			__syncthreads();
			rep = div_ceil (FiBiS_7 - it, maxit);
			for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			{
				int     j = ii;
				int     dp = (ii < FiSize_7) ? 0 : 2;
				double* train_loss = __loss + 1;
				double* p       = fb_w + FiBiP_7;
				double* g       = fl7_dw;
				double  decay_2 = l2_decay * __fdecay[DecayP_7 + dp    ];
				double  decay_1 = l1_decay * __fdecay[DecayP_7 + dp + 1];
				double* gsumi   = __gsumi + FiBiP_7;
				double* xsumi   = __xsumi + FiBiP_7;

				double p_j = p[j];

				double l2add = decay_2 * p_j * p_j / 2;		// loss_2
				double l1add = decay_1 * abs(p_j);			// loss_1
				train_loss[0] += l2add;
				train_loss[1] += l1add;
				//atomicAdd (train_loss + 0, l2add);
				//atomicAdd (train_loss + 1, l1add);

				double l2grad = decay_2 * (p_j);
				double l1grad = decay_1 * (p_j > 0 ? 1 : -1);

				double gij = (l2grad + l1grad + g[j]) / batch_size;

				if (momentum > 0)
				{
					double dx = momentum * gsumi[j] - learning_rate * gij;
					gsumi[j] = dx;
					p[j] += dx;
					//  atomicAdd (p + j, dx);						// p[j] += dx;
				}
				else
				{
					p[j] += -learning_rate * gij;
					//  atomicAdd (p + j, -learning_rate * gij);	// p[j] += -learning_rate * gij;
				}

				g[j] = 0;
			}

			__syncthreads();
			rep = div_ceil (FiBiS_10 - it, maxit);
			for (int c = 0, ii = it; c < rep; c++, ii += maxit)
			{
				int     j = ii;
				int     dp = (ii < FiSize_10) ? 0 : 2;
				double* train_loss = __loss + 1;
				double* p       = fb_w + FiBiP_10;
				double* g       = fl10_dw;
				double  decay_2 = l2_decay * __fdecay[DecayP_10 + dp    ];
				double  decay_1 = l1_decay * __fdecay[DecayP_10 + dp + 1];
				double* gsumi   = __gsumi + FiBiP_10;
				double* xsumi   = __xsumi + FiBiP_10;

				double p_j = p[j];

				double l2add = decay_2 * p_j * p_j / 2;		// loss_2
				double l1add = decay_1 * abs(p_j);			// loss_1
				train_loss[0] += l2add;
				train_loss[1] += l1add;
				//atomicAdd (train_loss + 0, l2add);
				//atomicAdd (train_loss + 1, l1add);

				double l2grad = decay_2 * (p_j);
				double l1grad = decay_1 * (p_j > 0 ? 1 : -1);

				double gij = (l2grad + l1grad + g[j]) / batch_size;

				if (momentum > 0)
				{
					double dx = momentum * gsumi[j] - learning_rate * gij;
					gsumi[j] = dx;
					p[j] += dx;
					//  atomicAdd (p + j, dx);						// p[j] += dx;
				}
				else
				{
					p[j] += -learning_rate * gij;
					//  atomicAdd (p + j, -learning_rate * gij);	// p[j] += -learning_rate * gij;
				}

				g[j] = 0;
			}
		}
		__syncthreads();
	}
}

}