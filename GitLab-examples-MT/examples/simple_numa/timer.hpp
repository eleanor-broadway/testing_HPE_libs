// Example codes for HPC course
// (c) 2012-2013 Andreas Hehn, ETH Zurich
#ifndef HPCSE_TIMER_HPP
#define HPCSE_TIMER_HPP

#include <chrono>

namespace hpcse
{

template <typename Clock = std::chrono::high_resolution_clock>
class timer {
  public:
    inline void start() {
        start_time = Clock::now();
    }
    inline void stop() {
        end_time   = Clock::now();
    }

    double get_timing() const {
        return std::chrono::nanoseconds(end_time - start_time).count()*1e-9;
    }
  private:
    std::chrono::time_point<Clock> start_time, end_time;
};

}

#endif // HPCSE_TIMER_HPP
