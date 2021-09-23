// Example codes for HPC course
// (c) 2012 Andreas Hehn, ETH Zurich

#ifndef HPCSE_NUMA_ALLOCATOR_HPP
#define HPCSE_NUMA_ALLOCATOR_HPP

#ifdef _WIN32
#include <malloc.h>
#endif
#include <cstdlib>

#if __cplusplus >= 201103L 
#define NOEXCEPT_SPEC noexcept
#else
#define NOEXCEPT_SPEC
#endif

namespace hpcse {

template <typename T>
class numa_allocator {
  public:
    typedef T*              pointer;
    typedef T const*        const_pointer;
    typedef T&              reference;
    typedef T const&        const_reference;
    typedef T               value_type;
    typedef std::size_t     size_type;
    typedef std::ptrdiff_t  difference_type;

    template <typename U>
    struct rebind {
        typedef numa_allocator<U> other;
    };

    numa_allocator() NOEXCEPT_SPEC {
    }

    numa_allocator(numa_allocator const& a) NOEXCEPT_SPEC {
    }

    template <typename U>
    numa_allocator(numa_allocator<U> const& b) NOEXCEPT_SPEC {
    }

    pointer allocate(size_type n) {
        void* p = malloc(n*sizeof(T));
        if(p == 0)
            throw std::bad_alloc();

        // NUMA first touch policy
        char * const pend = reinterpret_cast<char*>(p) + n*sizeof(T);
#pragma omp parallel for schedule(static)
        for(char* p1 = reinterpret_cast<char*>(p); p1 < pend; ++p1)
            *p1 = 0;
        return reinterpret_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type n) NOEXCEPT_SPEC {
        std::free(p);
    }

    size_type max_size() const NOEXCEPT_SPEC {
        std::allocator<T> a;
        return a.max_size();
    }

#if __cplusplus >= 201103L 
    template <typename C, class... Args>
    void construct(C* c, Args&&... args) {
        new ((void*)c) C(std::forward<Args>(args)...);
    }
#else
    void construct(pointer p, const_reference t) {
        new((void *)p) T(t);
    }
#endif

    template <typename C>
    void destroy(C* c) {
        c->~C();
    }

    bool operator == (numa_allocator const& a2) const NOEXCEPT_SPEC {
        return true;
    }

    bool operator != (numa_allocator const& a2) const NOEXCEPT_SPEC {
        return false;
    }

    template <typename U>
    bool operator == (numa_allocator<U> const& b) const NOEXCEPT_SPEC {
        return false;
    }

    template <typename U>
    bool operator != (numa_allocator<U> const& b) const NOEXCEPT_SPEC {
        return true;
    }
};

}

#undef NOEXPECT_SPEC

#endif //HPCSE_NUMA_ALLOCATOR_HPP
