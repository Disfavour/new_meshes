#include <vector>
#include <string>
#include <cassert>
#include <iostream>
#include "utility.h"
using namespace std;


bool sort_by_height(const rectangle& a, const rectangle& b) {
    return a.height > b.height;
}


bool sort_by_area(const rectangle& a, const rectangle& b) {
    return a.width * a.height > b.width * b.height;
}


void check_input(const size_t& unit_width, const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights) {
    assert(widths.size() == heights.size() && items_number == widths.size());

    for (auto& width : widths)
        assert(width <= unit_width);
}


vector<rectangle> convert_input(const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights) {
    vector<rectangle> rectangles;
    rectangles.reserve(items_number);
    for (size_t i = 0; i < items_number; i++)
        rectangles.push_back({widths[i], heights[i]});
    return rectangles;
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


void initialize_vectors_randomly(const size_t& unit_width, const size_t& items_number, vector<size_t>& widths, vector<size_t>& heights, const double& k) {
    assert(k > 0 && k <= 1);
    size_t max_rectangle_lenght = k * unit_width;
    assert(max_rectangle_lenght > 0 && max_rectangle_lenght <= unit_width);

    widths.reserve(items_number);
    heights.reserve(items_number);
    for (size_t i = 0; i < items_number; i++)
    {
        widths.push_back(rand() % max_rectangle_lenght + 1);
        heights.push_back(rand() % max_rectangle_lenght + 1);
    }
}