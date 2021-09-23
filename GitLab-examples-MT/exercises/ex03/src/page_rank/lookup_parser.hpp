#include <string>
#include <fstream>
#include <sstream>
#include <cassert>

std::string lookup_file;
std::size_t vector_size;

std::ifstream infile(lookup_file);
if(!infile) throw std::runtime_error("error reading lookup file");

std::vector<page_info> lookup_list;

size_type id; std::string page;
for (std::size_t i=0; i<vector_size; ++i) {
    infile >> id; infile >> std::ws;
    getline(infile, page, '\n');
    
    // ... do something with the page id (`id`) and the url (`page`)
    
}

infile.close();

