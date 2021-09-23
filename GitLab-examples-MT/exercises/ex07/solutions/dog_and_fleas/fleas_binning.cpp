/*****************************************************************************
*
* ALPS Project: Algorithms and Libraries for Physics Simulations
*
* ALPS Libraries
*
* Copyright (C) 2007 - 2010 by Matthias Troyer <troyer@comp-phys.org>
*
* This software is part of the ALPS libraries, published under the ALPS
* Library License; you can use, redistribute it and/or modify it under
* the terms of the license, either version 1 or (at your option) any later
* version.
* 
* You should have received a copy of the ALPS Library License along with
* the ALPS Libraries; see the file LICENSE.txt. If not, the license is also
* available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
*
*****************************************************************************/
// Adapted 2015 by Damian Steiger

#include "accumulator.hpp"
#include "fleas.hpp"

#include <iostream>
#include <fstream>
#include <random>
#include <valarray>

int main()
{
  const int N=50; // total number of fleas
  
  int M; // number of hops
  std::cout << "How many measurements? ";
  std::cin >> M;
  int Nhop; // number of hops
  std::cout << "How many hops between measurements? ";
  std::cin >> Nhop;
  unsigned int seed;
  std::cout << "Random number seed? ";
  std::cin >> seed;
  int n=N; // all fleas on left dog
  
  typedef std::mt19937 engine_type;
  typedef std::uniform_int_distribution<int> dist_type;  

  engine_type engine;
  engine.seed(seed);
  dist_type dist(1,N);

  // equilibration
  for (int i=0;i<M/5;++i) {
    if (dist(engine) <= n )
     --n;
    else
      ++n;
  }
  
  accumulator en;
  std::vector<accumulator> histogram (N+1,en);

  for (int i=0;i<M;++i) {
    for (int hop=0; hop < Nhop ; ++hop) {
      if (dist(engine) <= n )
       --n;
      else
        ++n;
    }
    
    for(int ii=0; ii<=50; ++ii) { 
      if(n!=ii) { 
        histogram[ii] << 0; 
      }
      else {
        histogram[ii] << 1;
      }
    }
  }
  
  for (int i=0;i<=N;++i)
    std::cout << i << "\t" 
              << probability(N,i) << "\t"
              << histogram[i] << "\n";
  return 0;
}
