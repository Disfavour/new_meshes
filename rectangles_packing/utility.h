#ifndef UTILITY_H
#define UTILITY_H

#include <vector>
#include <forward_list>
using namespace std;


struct rectangle {
    size_t width, height;
};


struct packed_rectangle {
    size_t x_min, x_max, y_min, y_max;
};


struct point {
    size_t x, y;
};


struct interval {
    size_t left, right;

    interval(size_t left, size_t right) : left(left), right(right) {}
};


struct task {
    size_t i, height;
    vector<forward_list<interval>> rows;
    vector<packed_rectangle> packed_items;
};


bool sort_by_height(const rectangle& a, const rectangle& b);
bool sort_by_area(const rectangle& a, const rectangle& b);

void check_input(const size_t& unit_width, const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights);
vector<rectangle> convert_input(const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights);
void show_solution(const size_t& unit_width, const size_t& max_height, const vector<packed_rectangle>& packed_rectangles);

void initialize_vectors_randomly(const size_t& unit_width, const size_t& items_number, vector<size_t>& widths, vector<size_t>& heights, const double& k=1);

#endif