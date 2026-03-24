#include <algorithm>
#include "utility.h"
#include "NFDH.h"


// NFDH (Next Fit Decreasing Height)
// rectangles must be sorted
size_t nfdh(const size_t& unit_width, const vector<rectangle>& rectangles, vector<packed_rectangle>& packed_rectangles) {
    packed_rectangles.clear();
    packed_rectangles.reserve(rectangles.size());
    size_t shelf_width = 0;
    size_t shelf_height = 0;
    size_t shelf_max_height = rectangles[0].height;
    for (const auto& rectangle : rectangles) {
        size_t potential_shelf_width = shelf_width + rectangle.width;

        if (potential_shelf_width > unit_width) {
            shelf_height = shelf_max_height;
            shelf_max_height += rectangle.height;
            shelf_width = 0;
            potential_shelf_width = rectangle.width;
        }

        packed_rectangles.push_back({shelf_width, potential_shelf_width, shelf_height, shelf_height + rectangle.height});
        shelf_width = potential_shelf_width;
    }
    return shelf_max_height;
}


size_t solve_problem_with_nfdh(const size_t& unit_width, const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights, vector<packed_rectangle>& packed_rectangles) {
    check_input(unit_width, items_number, widths, heights);

    vector<rectangle> rectangles{convert_input(items_number, widths, heights)};
    sort(rectangles.begin(), rectangles.end(), sort_by_height);

    size_t height = nfdh(unit_width, rectangles, packed_rectangles);

    show_solution(unit_width, height, packed_rectangles);

    return height;
}
