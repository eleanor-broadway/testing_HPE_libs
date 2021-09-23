#include <iostream>

using namespace std;

#define N 32
#define NT 16
#define NB 2
// reduction with 2 blocks of 16 each

__global__ void reduction(int * input, int * output)
{
	__shared__ int tmp[NT];
	
	tmp[threadIdx.x] = input[threadIdx.x + blockIdx.x * blockDim.x];

	__syncthreads();
	
	// 16 -> 8
	if (threadIdx.x < blockDim.x/2)
		tmp[threadIdx.x] += tmp[threadIdx.x + blockDim.x/2];
		
	__syncthreads();
	
	// 8 -> 4
	if (threadIdx.x < blockDim.x/4)
		tmp[threadIdx.x] += tmp[threadIdx.x + blockDim.x/4];
		
	__syncthreads();
	
	// 4 -> 2
	if (threadIdx.x < blockDim.x/8)
		tmp[threadIdx.x] += tmp[threadIdx.x + blockDim.x/8];
		
	__syncthreads();
	
	// 2 -> 1
	if (threadIdx.x == 0)
	{
		tmp[threadIdx.x] += tmp[threadIdx.x + 1];
		output[blockIdx.x] = tmp[threadIdx.x];
	}
}


int main()
{
	int h_input[N], h_output[NB];
	int * d_input, * d_output;
	
	for (int i=0; i<N; i++)
		h_input[i] = 1;
		
	cudaMalloc( (void**)&d_input, N*sizeof(int) );
	cudaMalloc( (void**)&d_output, NB*sizeof(int) );
	
	cudaMemcpy( d_input, h_input, N*sizeof(int), cudaMemcpyHostToDevice );
	
	reduction<<< NB,NT >>>(d_input, d_output);
	
	cudaMemcpy( h_output, d_output, NB*sizeof(int), cudaMemcpyDeviceToHost );
	
	cout << "Result0 is " << h_output[0] << endl;
	cout << "Result1 is " << h_output[1] << endl;

	return 0;
}