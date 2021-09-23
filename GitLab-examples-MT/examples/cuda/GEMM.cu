#include <iostream>

#define N 16

using namespace std;

__global__ void matrix(float * a, float * b, float * c)
{
	int ix = threadIdx.x + blockIdx.x*blockDim.x;
	int iy = threadIdx.y + blockIdx.y*blockDim.y;
	
	if (ix<N && iy<N)
	{
		c[ix*N + iy] = 0;
		
		for (int k=0; k<N; k++)
			c[ix*N + iy] += a[ix*N + k] * b[k*N + iy];
	}
	
}


// ---------------------------------------------------------  Main function  -----------------------------------------------------------			   


int main () 
{ 
	// Allocate memory on the CPU
	float * a = new float[N*N];
	float * b = new float[N*N];
	float * c = new float[N*N];
	
	// Declare variables on the device (GPU)
	float * dev_a;
	float * dev_b;
	float * dev_c;
	
	// Allocate memory on the GPU
	cudaMalloc( (void**)&dev_a, N * N * sizeof(float) );
	cudaMalloc( (void**)&dev_b, N * N * sizeof(float) ); 
	cudaMalloc( (void**)&dev_c, N * N * sizeof(float) );
	
	// Fill the matrices a , b 
	for (int i=0; i<N; i++)
		for (int j=0; j<N; j++)
		{		
			a[i*N + j] = 1;
			b[i*N + j] = 2;
		}
	
    // Copy the arrays 'a' and 'b' to the GPU
	cudaMemcpy( dev_a, a, N * N * sizeof(float), cudaMemcpyHostToDevice );
    cudaMemcpy( dev_b, b, N * N * sizeof(float), cudaMemcpyHostToDevice ); 
	
    // Call the multiplication function
	dim3 blocks(N/4,N/4);
	dim3 threads(4,4); 
	matrix<<< blocks, threads >>>(dev_a, dev_b, dev_c);
	
	// Copy the array 'c' from the GPU back to the CPU
	cudaMemcpy( c, dev_c, N * N * sizeof(float), cudaMemcpyDeviceToHost );
	
	// Print the results
	for (int i=0; i<N; i++)
	{
		for (int j=0; j<N; j++)
			cout << c[i*N + j] << " ";
		cout << endl; 
	}
	
	// Free the memory allocated on the GPU
    cudaFree( dev_a );
	cudaFree( dev_b ); 
	cudaFree( dev_c );
	
	return 0;
}




