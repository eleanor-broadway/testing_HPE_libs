#include <iostream>

int main(int argc, char* argv[])
{
    cudaDeviceProp dev_prop;
    int dev_cnt = 0;
    cudaGetDeviceCount(&dev_cnt);
    for(int i=0; i < dev_cnt; ++i)
    {
        cudaGetDeviceProperties(&dev_prop, i);
        std::cout << "Device : " << i << " has compute capability " << dev_prop.major << "." << dev_prop.minor << std::endl;
    }
    return 0;
}
