#include <iostream>

#define N 500 

using namespace std;

__global__ void dot( int *a, int *b, int *c )
{
	// Shared memory for results of multiplication	
	__shared__ int temp[N]; 
	temp[threadIdx.x] = a[threadIdx.x] * b[threadIdx.x];
	
	__syncthreads();
	
	// Thread 0 sums the pairwise products
	if( 0 == threadIdx.x ) 
	{
		int sum = 0;
		for( int i = 0; i < N; i++ )
			sum += temp[i];
	
		*c = sum;
	}
}

int main()
{
	int h_a[N], h_b[N], h_c;
	int *d_a, *d_b, *d_c;
	
	cudaMalloc( (void**) &d_a, N*sizeof(int) );
	cudaMalloc( (void**) &d_b, N*sizeof(int) );
	cudaMalloc( (void**) &d_c, sizeof(int) );
	
	for (int i=0; i<N; i++)
	{
		h_a[i] = 1;
		h_b[i] = i;
	}
	
	cudaMemcpy( d_a, h_a, N*sizeof(int), cudaMemcpyHostToDevice );
	cudaMemcpy( d_b, h_b, N*sizeof(int), cudaMemcpyHostToDevice );
	
	dot<<< 1,N >>>(d_a, d_b, d_c);
	
	cudaMemcpy( &h_c, d_c, sizeof(int), cudaMemcpyDeviceToHost );
	
	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	
	cout << h_c << endl;
	
	return 0;
}