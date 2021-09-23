#include <vector>
#include <iostream>
#include <cmath>

__global__ void vector_add_kernel(float * r, float * v1, float * v2, int size)
{
    //
    // TODO
    // complete kernel
    //
}


void vector_add_cpu(float * r, float * v1, float * v2, int size)
{
    for(std::size_t i=0; i < size; ++i)
    {
        r[i] = v1[i] + v2[i];
    }
}

void fill_vector(std::vector<float> & v)
{
    for(std::size_t i=0; i < v.size(); ++i)
    {
        v[i] = i;
    }
}

bool compare_vectors(std::vector<float> const& v1, std::vector<float> const& v2)
{
    if(v1.size() != v2.size())
    {
        std::cout << "ERROR: Vector sizes mismatch!" << std::endl;
        return false;
    }
    bool ok = true;
    for(std::size_t i=0; i < v1.size(); ++i)
    {
        if(std::abs(v1[i]-v2[i]) > 1e-5)
        {
            std::cout << "ERROR: element " << i << " mismatch: " << v1[i] << " != " << v2[i] << std::endl;
            ok = false;
        }
    }
    return ok;
}

int main()
{
    int const N = 1000;
    std::vector<float> a(N);
    std::vector<float> b(N);
    std::vector<float> c(N);

    fill_vector(a);
    fill_vector(b);

    vector_add_cpu(&c[0], &a[0], &b[0], N);

    // Create copies on GPU device
    //  Allocate memory
    float * d_a;
    float * d_b;
    float * d_c;
    cudaMalloc(&d_a, N*sizeof(float)); 
    cudaMalloc(&d_b, N*sizeof(float)); 
    cudaMalloc(&d_c, N*sizeof(float)); 
    //  Copy
    cudaMemcpy(d_a, &a[0], N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, &b[0], N*sizeof(float), cudaMemcpyHostToDevice);

    //
    // TODO
    // Kernel call
    //

    // Get result
    std::vector<float> c_from_gpu(N);
    cudaMemcpy(&c_from_gpu[0], d_c, N*sizeof(float), cudaMemcpyDeviceToHost);

    bool ok = compare_vectors(c, c_from_gpu);

    // Free the device memory
    cudaFree(d_c);
    cudaFree(d_b);
    cudaFree(d_a);

    if(ok)
        std::cout << "Results match... It works!" << std::endl;
    return ok ? 0 : 1;
}
