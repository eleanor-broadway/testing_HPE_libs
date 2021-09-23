#include <iostream>
#include <cassert>

#define N 32768

__global__ void scaleVector(float scale, float * input, float * output)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	if (tid < N)
	{
		output[tid] = input[tid] * scale;
	}
}

int main()
{
	float * h_a = new float[N];
	float * h_b = new float[N];

	float * d_a;
	float * d_b;
	
	const float scale = 2.;

	for (int i=0; i<N; i++)
		h_a[i] = (float)i/2.;
	
	std::cout << "Initializing data on GPU\n";

	cudaMalloc( (void**)&d_a, N*sizeof(float) );
	cudaMalloc( (void**)&d_b, N*sizeof(float) );

	cudaMemcpy( d_a, h_a, N*sizeof(float), cudaMemcpyHostToDevice );


	std::cout << "Launching kernels on GPU\n";

	const int nblocks = 128;
	const int nthreads = 256;
	scaleVector<<< nblocks, nthreads >>>(scale, d_a, d_b);

	std::cout << "Downloading data\n";

	cudaMemcpy( h_b, d_b, N*sizeof(float), cudaMemcpyDeviceToHost );

	std::cout << "Verifying results\n";

	for (int i=0; i<N; i++)
	{
		std::cout << h_b[i] << std::endl;
		assert((double)i == h_b[i]);
	}

	std::cout << "Done!\n";

	cudaFree(d_a);
	cudaFree(d_b);

	delete [] h_a;
	delete [] h_b;
}
