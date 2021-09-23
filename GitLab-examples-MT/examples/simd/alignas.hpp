// Example codes for HPC course
// (c) 2012 Matthias Troyer, ETH Zurich

// workaround for missing alignas in g++-4.7 and MSVC
#ifndef HPC12_ALIGNAS_HPP
#define HPC12_ALIGNAS_HPP

#if defined(_MSC_VER) and _MSC_VER<=1700
#define alignas(A) __declspec(align(A))
#elif (__GNUC__==4) and (__GNUC_MINOR__ <= 7) 
#define alignas(A) __attribute__((aligned(A)))
#endif

#endif
