#include "utility.h"
#include "NFDH.h"
#include "skyline.h"
#include "exact.h"

int main() {
    size_t unit_width = 10;
    size_t items_number = 20; //5 18 20
    vector<size_t> widths;
    vector<size_t> heights;

    initialize_vectors_randomly(unit_width, items_number, widths, heights, 0.5);

    vector<packed_rectangle> packed_rectangles;
    solve_problem_with_nfdh(unit_width, items_number, widths, heights, packed_rectangles);
    solve_problem_with_skyline(unit_width, items_number, widths, heights, packed_rectangles);
    solve_problem_exact(unit_width, items_number, widths, heights, packed_rectangles);

    // packed_rectangles.clear();
    // vector<rectangle> rectangles{convert_input(items_number, widths, heights)};
    // size_t height = exact_recursive(unit_width, rectangles, packed_rectangles, 20);
    // show_solution(unit_width, height, packed_rectangles);

    return 0;
}