extern "C"
{   
// loss[0] : l2_decay
// loss[1] : l1_decay

__global__ void sgd_nm (double* loss, double* p, double* g, 
						double l2_decay, double l1_decay, int batch_size, double learning_rate)
{
	int j; double l1grad; double l2grad; double gij;
	
	j = blockDim.x * blockIdx.x + threadIdx.x;

	//if (j < outN)
	{
		double l2add = l2_decay * p[j] * p[j] / 2; // loss_2
		double l1add = l1_decay * abs(p[j]);       // loss_1
		atomicAdd (loss + 0, l2add);
		atomicAdd (loss + 1, l1add);

		l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
		l2grad = l2_decay * (p[j]);

		gij = (l2grad + l1grad + g[j]) / batch_size;

		p[j] += -learning_rate * gij;

		g[j] = 0;
	}
}
__global__ void sgd_mome (double* loss, double* p, double* g, double* gsumi,
						  double l2_decay, double l1_decay, int batch_size, double learning_rate, double momentum)
{
	int j; double l1grad; double l2grad; double gij; double dx;

	j = blockDim.x * blockIdx.x + threadIdx.x;

	//if (j < outN)
	{
		double l2add = l2_decay * p[j] * p[j] / 2; // loss_2
		double l1add = l1_decay * abs(p[j]);       // loss_1
		atomicAdd (loss + 0, l2add);
		atomicAdd (loss + 1, l1add);

		l1grad = l1_decay * (p[j] > 0 ? 1 : -1);
		l2grad = l2_decay * (p[j]);

		gij = (l2grad + l1grad + g[j]) / batch_size;

		dx = momentum * gsumi[j] - learning_rate * gij;
		gsumi[j] = dx;
		p[j] += dx;

		g[j] = 0;
	}
}

}
