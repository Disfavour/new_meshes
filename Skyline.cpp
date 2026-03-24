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

struct point {
    size_t x, y;
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


packed_rectangle place_item(const rectangle& item, const vector<point>& skyline, const size_t& unit_width) {
    packed_rectangle packed_item;

    size_t x_max = skyline[0].x + item.width;
    size_t y_max = skyline[0].y;
    for (size_t j = 1; j < skyline.size(); j++)
    {
        if (skyline[j].x > x_max)
            break;
        
        y_max = max(y_max, skyline[j].y);
    }
    packed_item.y_min = y_max;
    packed_item.x_min = skyline[0].x;

    for (size_t i = 1; i < skyline.size(); i++)
    {
        x_max = skyline[i].x + item.width;
        if (x_max > unit_width)
            break;
        
        y_max = skyline[i].y;
        for (size_t j = i + 1; j < skyline.size(); j++)
        {
            if (skyline[j].x >= x_max)
                break;
            
            y_max = max(y_max, skyline[j].y);
        }

        if (packed_item.y_min > y_max) {
            packed_item.y_min = y_max;
            packed_item.x_min = skyline[i].x;
        }
    }
    
    packed_item.x_max = packed_item.x_min + item.width;
    packed_item.y_max = packed_item.y_min + item.height;

    return packed_item;
}


void update_skyline(const packed_rectangle& packed_item, vector<point>& skyline, vector<point>& old_skyline) {
    old_skyline = skyline;
    skyline.clear();
    point left{packed_item.x_min, packed_item.y_max};
    size_t x_max_height = 0;
    size_t previous_height = 0;
    bool is_inserted_after = false;

    for (const auto& p : old_skyline) {
        if (p.x < packed_item.x_min) {
            skyline.push_back(p);
        }
        else if (p.x == packed_item.x_min && packed_item.y_max != previous_height) {
            skyline.push_back({p.x, packed_item.y_max});
        }
        else if (p.x == packed_item.x_max) {
            if (p.y != packed_item.y_max) {
                skyline.push_back(p);
            }
            is_inserted_after = true;
        }
        else if (p.x > packed_item.x_max) {
            if (!is_inserted_after) {
                skyline.push_back({packed_item.x_max, previous_height});
                is_inserted_after = true;
            }
            skyline.push_back(p);
        }

        previous_height = p.y;
    }
    if (!is_inserted_after) {
        skyline.push_back({packed_item.x_max, previous_height});
        is_inserted_after = true;
    }

    for (size_t i = 1; i < skyline.size(); i++)
    {
        assert(skyline[i-1].y != skyline[i].y);
    }
}


// Skyline (Bottom-Left Skyline with Min Y)
vector<packed_rectangle> solve_problem(const size_t& unit_width, const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights) {
    check_input(unit_width, items_number, widths, heights);

    vector<rectangle> rectangles;
    rectangles.reserve(items_number);
    for (size_t i = 0; i < items_number; i++)
        rectangles.push_back({widths[i], heights[i]});
    
    sort(rectangles.begin(), rectangles.end(), sort_by_height);

    vector<packed_rectangle> packed_rectangles;
    packed_rectangles.reserve(items_number);
    size_t max_height = 0;
    vector<point> skyline{{0, 0}}, old_skyline;
    for (const auto& rectangle : rectangles) {
        packed_rectangle packed_item(place_item(rectangle, skyline, unit_width));
        packed_rectangles.push_back(packed_item);
        max_height = max(max_height, packed_item.y_max);

        update_skyline(packed_item, skyline, old_skyline);
    }

    show_solution(unit_width, max_height, packed_rectangles);

    return packed_rectangles;
}


void initialize_vectors_randomly(const size_t& unit_width, const size_t& items_number, vector<size_t>& widths, vector<size_t>& heights) {
    widths.reserve(items_number);
    heights.reserve(items_number);
    for (size_t i = 0; i < items_number; i++)
    {
        widths.push_back(rand() % (unit_width / 5) + 1);
        heights.push_back(rand() % (unit_width / 5) + 1);
    }
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

    initialize_vectors_randomly(unit_width, items_number, widths, heights);
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
