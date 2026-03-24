#ifndef skyline_H
#define skyline_H

#include "utility.h"

size_t skyline(const size_t& unit_width, const vector<rectangle>& rectangles, vector<packed_rectangle>& packed_rectangles);
size_t solve_problem_with_skyline(const size_t& unit_width, const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights, vector<packed_rectangle>& packed_rectangles);

#endif