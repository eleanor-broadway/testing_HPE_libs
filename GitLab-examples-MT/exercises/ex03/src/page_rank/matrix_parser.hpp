#include <string>
#include <fstream>
#include <sstream>
#include <cassert>

std::string matrix_file;

std::ifstream infile(matrix_file);
if(!infile) throw std::runtime_error("error reading matrix file");

std::string line;
for(size_type i = 0; i < 2; ++i) std::getline(infile,line,'\n'); //skip file headers

/// fetch matrix dimensions: number of rows,
/// number of columns and number of non-zero elements
size_type n_rows, n_cols, nnz_tot;
infile >> n_rows >> n_cols >> nnz_tot;
if (n_rows != n_cols) throw std::runtime_error("matrix is not square");

/// read matrix from file: row index, column index, value
size_type row, col; value_type val;
while (infile >> row >> col >> val) {
    // ... do something with row, col and val
    
}

infile.close();
