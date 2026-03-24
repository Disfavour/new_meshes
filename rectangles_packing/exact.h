#ifndef exact_H
#define exact_H

#include "utility.h"

size_t exact_recursive(const size_t& unit_width, vector<rectangle>& rectangles, vector<packed_rectangle>& packed_rectangles, const size_t& best_height);
size_t exact(const size_t& unit_width, vector<rectangle>& rectangles, vector<packed_rectangle>& packed_rectangles, const size_t& best_height);
size_t solve_problem_exact(const size_t& unit_width, const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights, vector<packed_rectangle>& packed_rectangles);

#endif