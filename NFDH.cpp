#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <cassert>
using namespace std;


struct rectangle {
    size_t width, height;
};


struct packed_rectangle {
    size_t x_min, x_max, y_min, y_max;
};


void check_input(const size_t& unit_width, const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights) {
    assert(widths.size() == heights.size() && items_number == widths.size());

    for (auto& width : widths)
        assert(width <= unit_width);
}


bool sort_by_height(const rectangle& a, const rectangle& b) {
    return a.height > b.height;
}


void show_solution(const size_t& unit_width, const size_t& max_height, const vector<packed_rectangle>& packed_rectangles) {
    vector<string> image(max_height);
    for (auto& row : image) {
        row.reserve(unit_width);
        for (size_t j = 0; j < unit_width; j++)
            row.push_back(' ');
    }
    
    string symbols{"!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"};
    assert(packed_rectangles.size() <= symbols.size());
    
    for (size_t i = 0; i < packed_rectangles.size(); i++)
        for (size_t j = packed_rectangles[i].y_min; j < packed_rectangles[i].y_max; j++)
        {
            assert(packed_rectangles[i].x_max <= unit_width);
            for (size_t k = packed_rectangles[i].x_min; k < packed_rectangles[i].x_max; k++)
            {
                assert(image[j][k] == ' ');
                image[j][k] = symbols[i];
            }
        }
            
                
    for (auto it = image.rbegin(); it != image.rend(); ++it)
        cout << '|' << *it << '|' << endl;
    
    for (size_t i = 0; i < unit_width + 2; i++)
    {
        cout << '-';
    }
    cout << endl << "Max height " << max_height << endl;
}


// NFDH (Next Fit Decreasing Height)
vector<packed_rectangle> solve_problem(const size_t& unit_width, const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights) {
    check_input(unit_width, items_number, widths, heights);

    vector<rectangle> rectangles;
    rectangles.reserve(items_number);
    for (size_t i = 0; i < items_number; i++)
        rectangles.push_back({widths[i], heights[i]});
    
    sort(rectangles.begin(), rectangles.end(), sort_by_height);

    vector<packed_rectangle> packed_rectangles;
    packed_rectangles.reserve(items_number);
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

    show_solution(unit_width, shelf_max_height, packed_rectangles);

    return packed_rectangles;
}

void initialize_1(size_t& unit_width, size_t& items_number, vector<size_t>& widths, vector<size_t>& heights) {
    unit_width = 10;
    items_number = 6;
    widths = {2, 6, 4, 3, 5, 8};
    heights = {2, 4, 4, 3, 3, 2};
}


void initialize_2(size_t& unit_width, size_t& items_number, vector<size_t>& widths, vector<size_t>& heights) {
    unit_width = 50;
    items_number = 94;

    widths.reserve(items_number);
    heights.reserve(items_number);
    for (size_t i = 0; i < items_number; i++)
    {
        widths.push_back(rand() % (unit_width / 5) + 1);
        heights.push_back(rand() % (unit_width / 5) + 1);
    }
}


int main() {
    size_t unit_width;
    size_t items_number;
    vector<size_t> widths;
    vector<size_t> heights;

    initialize_2(unit_width, items_number, widths, heights);

    vector<packed_rectangle> packed_rectangles(solve_problem(unit_width, items_number, widths, heights));

    return 0;
}