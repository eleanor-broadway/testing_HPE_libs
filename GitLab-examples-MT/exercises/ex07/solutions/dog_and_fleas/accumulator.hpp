// Example codes for HPC course
// (c) ALPS Project, http://alps.comp-phys.org
// (c) Adapted by Michele Dolfi, ETH Zurich, 2012
// (c)            Jan Gukelberger, ETH Zurich, 2012
// (c)            Matthias Troyer, ETH Zurich, 2012
// (c)            Damian Steiger, ETH Zurich, 2015

#ifndef HPC12_ACCUMULATOR_H
#define HPC12_ACCUMULATOR_H

#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

/* Example:
 * 
 * // initialization of accumulator
 * accumulator en;
 *
 * // pushing some values
 * en << 1.;
 * en << 42.;
 * ...
 *
 * // reading out statistics
 * std::cout << "Energy = " << en.mean() << "+/-" << en.error();
 * std::cout << "Tau = " << en.tau();
 * // or simply use the cout overload
 * std::cout << en;
 * 
 * // save binning analysis (header <fstream> needed)
 * std::ofstream of("binning.dat");
 * en.save_binning_analysis(of);
 *
 */

class accumulator {
public:
    typedef double             value_type;
    typedef double             time_type;
    typedef std::size_t        size_type;
    
    accumulator()
    : count_(0)
    { }
    
    void operator<<(value_type const& x)
    {
        // set sizes if starting additions
        if(count_ == 0)
        {
            sum_.resize(1);
            sum2_.resize(1);
            bin_entries_.resize(1);
        }
        
        // store x, x^2
        sum_[0]     += x;
        sum2_[0]    += x*x;
        
        size_type i=count_;
        count_++;
        bin_entries_[0]++;
        size_type binlen=1;
        size_type bin=0;
        
        // binning
        do
        {
            if(i&1)
            {
                // a bin is filled
                binlen*=2;
                bin++;
                if(bin>=bin_entries_.size())
                {
                    sum_.resize(std::max(bin+1, sum_.size()));
                    sum2_.resize(std::max(bin+1,sum2_.size()));
                    bin_entries_.resize(std::max(bin+1,bin_entries_.size()));
                }
                
                value_type x1=(sum_[0]-sum_[bin]);
                x1/=double(binlen);
                
                value_type y1 = x1*x1;
                
                sum2_[bin] += y1;
                sum_[bin] = sum_[0];
                bin_entries_[bin]++;
            }
            else
                break;
        } while ( i>>=1);
    }
    
    size_type count() const
    {
        return count_;
    }
    
    size_type binning_depth() const
    {
        return ( int(sum_.size())-7 < 1 ) ? 1 : int(sum_.size())-7;
    }
    
    value_type mean() const
    {
        if (count()==0)
            throw std::runtime_error("No measurement.");
        return sum_[0]/double(count());
    }
    
    value_type variance() const
    {
        if (count()==0)
            throw std::runtime_error("No measurement.");
        
        if(count()<2)
            return std::numeric_limits<double>::infinity();
        
        value_type var = (sum2_[0] - sum_[0] * sum_[0]/count())/(count()-1);
        return (var < 0.) ? 0. : var;
    }
    
    // error estimated from bin i, or from default bin if <0
    value_type error(std::size_t i=std::numeric_limits<std::size_t>::max()) const
    {
        if (count()==0)
            throw std::runtime_error("No measurement.");
        
        if (i==std::numeric_limits<std::size_t>::max())
            i=binning_depth()-1;
        
        if (i > binning_depth()-1)
            throw std::invalid_argument("Invalid bin.");
        
        size_type binsize = bin_entries_[i];
        
        return std::sqrt( binvariance(i) / double(binsize-1) );
    }
    
    time_type tau() const
    {
        if (count()==0)
            throw std::runtime_error("No measurement.");
        
        if( binning_depth() >= 2 )
        {
            time_type er = error();
            return 0.5*( er*er*count() / variance() - 1. );
        } else {
            return std::numeric_limits<double>::infinity();
        }
    }
    
    void save_binning_analysis(std::ostream & os) const
    {
        os << "# Level\terror" << std::endl;
        for (size_type k=0; k<binning_depth(); ++k)
            os << k << "\t" << error(k) << std::endl;
    }
    
private:
    
    value_type binmean(size_type i) const
    {
        return sum_[i]/(bin_entries_[i] * (1ll<<i));
    }
    
    value_type binvariance(size_type i) const
    {
        value_type retval(sum2_[i]);
        retval/=double(bin_entries_[i]);
        retval-=binmean(i)*binmean(i);
        // without prefactor M/(M-1)
        return retval;
    }
    
    std::vector<value_type> sum_;        // sum of measurements in the bin
    std::vector<value_type> sum2_;       // sum of the squares
    std::vector<size_type> bin_entries_; // number of measurements
    size_type count_; // total number of measurements (=bin_entries_[0])
};


// OVERLOADS

inline std::ostream& operator<<(std::ostream& os, accumulator const & obs)
{
    if(obs.count())
    {
        os << std::setprecision(6) << obs.mean() << " +/- "
           << std::setprecision(3) << obs.error() << "; tau = "
           << std::setprecision(3) << ((std::abs(obs.error())>1e-12) ? obs.tau() : 0)
           << std::setprecision(6);
    }
    return os;
}

#endif // HPC12_ACCUMULATOR_H
