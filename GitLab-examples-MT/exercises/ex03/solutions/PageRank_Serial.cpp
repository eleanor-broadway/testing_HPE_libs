#include <iostream>
#include <algorithm>
#include <functional>
#include <iterator>
#include <string>
#include <fstream>
#include <sstream>
#include <cassert>
#include <vector>
#include <chrono>
#include <stdexcept>

typedef double value_type;
typedef std::size_t size_type;

class csr_matrix
{
    
public:
    
    csr_matrix(const std::string& matrix_file)
    {
        std::ifstream infile(matrix_file);
        if(!infile) throw std::runtime_error("error reading matrix file");
        
        std::string line;
        for(size_type i = 0; i < 2; ++i) std::getline(infile,line,'\n'); //skip file headers
        
        // fetch matrix dimensions
        size_type n_rows, n_cols, nnz_tot;
        infile >> n_rows >> n_cols >> nnz_tot;
        if (n_rows != n_cols) throw std::runtime_error("matrix is not square");
        
        //read matrix from file
        data_.reserve(nnz_tot);
        col_indices_.reserve(nnz_tot);
        
        row_starts_.resize(n_rows+1);
        row_starts_[0] = 0;
        
        size_type last_row = 0;
        size_type row, col; value_type val;
        while (infile >> row >> col >> val) {
            if(row == last_row + 1) {
                row_starts_[row] = data_.size();
                last_row = row;
            }
            else if(row > last_row + 1) { // there are empty rows
                std::fill(row_starts_.begin()+last_row+1, row_starts_.begin()+row, data_.size());
                row_starts_[row] = data_.size();
                last_row = row;
            }
            
            col_indices_.push_back(col);
            data_.push_back(val);
            
            assert(last_row < n_rows);
        }
        
        infile.close();
        
        std::fill(row_starts_.begin()+last_row+1, row_starts_.end(), data_.size());
    }
    
    inline value_type data(const size_type i) const
    { return data_[i]; }
    
    inline size_type col_index(const size_type i) const
    { return col_indices_[i]; }
    
    inline size_type row_begin(const size_type i) const
    { return row_starts_[i]; }
    
    inline size_type row_end(const size_type i) const
    { return row_starts_[i+1]; }
    
    inline size_type num_rows() const
    { return row_starts_.size() - 1; }
    
    inline size_type num_non_zero_elements() const
    { return data_.size(); }
    
private:
    
    std::vector<value_type> data_; //size: nnz_tot
    std::vector<size_type> col_indices_; //size: nnz_tot
    std::vector<size_type> row_starts_; //size: n_rows+1
};

struct page_info
{
    size_type id;
    std::string page;
    value_type page_rank;
    
    bool operator>(const page_info& b) const {return this->page_rank > b.page_rank;}
    
    friend std::ostream& operator<< (std::ostream &out, const page_info &PI);
};

std::ostream& operator<< (std::ostream &out, const page_info &PI)
{
    out
    << "("
    << PI.id << "\t"
    << PI.page << "\t"
    << PI.page_rank
    << ")";
    
    return out;
}

std::vector<page_info> get_lookup_list(const std::vector<value_type>& page_rank, const std::string& lookup_file)
{
    std::ifstream infile(lookup_file);
    if(!infile) throw std::runtime_error("error reading lookup file");
    
    std::vector<page_info> lookup_list;
    lookup_list.reserve(page_rank.size());
    
    size_type id; std::string page;
    for (value_type prank : page_rank) {
        infile >> id; infile >> std::ws;
        getline(infile, page, '\n');
        lookup_list.push_back( page_info{id, page, prank} );
    }

    infile.close();

    return lookup_list;
}

int main(int argc, char* argv[])
{
    using std::swap;
    using std::sqrt;
    using std::abs;
    std::cout << "Inits... " << std::endl;
    
    const std::string domain = "ETH_network";
    //const std::string domain = "US_patents";
    
    const std::string matrix_file = domain + ".mtx";
    const std::string lookup_file = domain + ".lst";
    
    std::cout
              << "matrix file : " << matrix_file << "\n"
              << "lookup file : " << lookup_file << "\n";
    
    const value_type p_ = domain == "ETH_network" ? 0.3 : 0.1;
    
    //allocate page rank vector and fill CSR matrix
    
    const csr_matrix prm(matrix_file);
    
    const size_type pr_size = prm.num_rows();
    
    std::vector<value_type> page_rank(pr_size,1./pr_size);
    
    std::cout
              << "matrix size: " << pr_size << "\n"
              << "non-zero elements: " << prm.num_non_zero_elements() << "\n"
              << "p: " << p_ << "\n";
    
    std::cout << "\n";
    
    //calculate page rank
    
    std::cout << "Calculating page rank... " << std::endl;
    
    size_type iter = 0;
    bool converged = false;
    std::vector<value_type> page_rank_tmp(pr_size);
    
    std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
    start = std::chrono::high_resolution_clock::now();
    
    value_type norm = 0;
    value_type sum  = 0;
    for(value_type x : page_rank)
    {
        sum  += x;
        norm += x*x;
    }
    norm = sqrt(norm);
    while(!converged){
        
        value_type const prx = sum * p_ / pr_size; // result of constant matrix
        value_type sum_tmp  = 0;
        value_type norm_tmp = 0;
        value_type theta    = 0;
        for(size_type i = 0; i < pr_size; ++i){
            
            value_type x = 0;
            for(size_type el = prm.row_begin(i); el < prm.row_end(i); ++el)
                x += page_rank[prm.col_index(el)] * prm.data(el);

            x = ((1-p_) * x + prx) / norm;
            page_rank_tmp[i]  = x;
            sum_tmp          += x;
            norm_tmp         += x*x;
            theta            += x * page_rank[i] / norm;
        }
        norm_tmp = sqrt(norm_tmp);

        // convergence check
        for(size_type i = 0; i < pr_size; ++i)
            page_rank[i] = page_rank_tmp[i] - theta * page_rank[i]/norm;

        value_type conv = 0;
        for(value_type x : page_rank)
            conv += x*x;
        conv = sqrt(conv);
        conv /= abs(theta);
        converged = conv < 1e-6;

        //swap in new vector
        swap(page_rank, page_rank_tmp);
        swap(norm, norm_tmp);
        swap(sum, sum_tmp);
        
        std::cout << iter << '\t' << conv << std::endl;
        iter++;
    }

    // Normalize results
    for(value_type& x : page_rank)
        x /= norm;
    
    end = std::chrono::high_resolution_clock::now();
    
    double elapsed = std::chrono::duration<double>(end-start).count();
    
    std::cout << "\n";
    std::cout << "exec time: " << '\t' << elapsed << std::endl;
    std::cout << "\n";
    
    //build page list according to page rank
    
    std::cout << "Building page list.. " << std::endl;
    
    std::vector<page_info> lookup_list = get_lookup_list(page_rank,lookup_file);
    std::sort(lookup_list.begin(), lookup_list.end(), std::greater<page_info>());
    
    for(size_type i = 0; i < 20; ++i)
        std::cout << lookup_list[i] << std::endl;
    
    return 0;
}
