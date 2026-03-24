#include <cassert>
#include <algorithm>
#include "utility.h"
#include "skyline.h"


packed_rectangle place_item(const rectangle& item, const vector<point>& skyline, const size_t& unit_width) {
    packed_rectangle packed_item;

    size_t x_max = skyline[0].x + item.width;
    size_t y_max = skyline[0].y;
    for (size_t j = 1; j < skyline.size(); j++)
    {
        if (skyline[j].x >= x_max)
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
// rectangles must be sorted
size_t skyline(const size_t& unit_width, const vector<rectangle>& rectangles, vector<packed_rectangle>& packed_rectangles) {
    packed_rectangles.clear();
    packed_rectangles.reserve(rectangles.size());
    size_t max_height = 0;
    vector<point> skyline{{0, 0}}, old_skyline;
    for (const auto& rectangle : rectangles) {
        packed_rectangle packed_item(place_item(rectangle, skyline, unit_width));
        packed_rectangles.push_back(packed_item);
        max_height = max(max_height, packed_item.y_max);

        update_skyline(packed_item, skyline, old_skyline);
    }
    return max_height;
}


size_t solve_problem_with_skyline(const size_t& unit_width, const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights, vector<packed_rectangle>& packed_rectangles) {
    check_input(unit_width, items_number, widths, heights);

    vector<rectangle> rectangles{convert_input(items_number, widths, heights)};
    sort(rectangles.begin(), rectangles.end(), sort_by_height);

    size_t height = skyline(unit_width, rectangles, packed_rectangles);

    show_solution(unit_width, height, packed_rectangles);

    return height;
}
