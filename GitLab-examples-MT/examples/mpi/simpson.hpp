// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

#include <cassert>
#include <functional>

inline double simpson(double (*f) (double), double a, double b, unsigned int N)
{
    assert (b>=a);
    assert (N!=0u);
    
    double       h=(b-a)/N;
    
    // boundary values
    double result = ( f(a) + 4*f(a+h/2) + f(b) ) / 2.0;
    
    // values between boundaries
    for ( unsigned int i = 1; i <= N-1; ++i )
        result += f(a+i*h) + 2*f(a+(i+0.5)*h);
        
    return result * h / 3.0;
}
