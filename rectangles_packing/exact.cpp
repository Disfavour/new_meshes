#include <algorithm>
#include <cmath>
#include <numeric>
#include <iostream>
#include <stack>
#include <cassert>
#include "utility.h"
#include "NFDH.h"
#include "skyline.h"


size_t min_height_estimation(const size_t& unit_width, const vector<rectangle>& rectangles) {
    size_t full_area = accumulate(
        rectangles.begin(),
        rectangles.end(), 
        0,
        [](size_t sum, const rectangle& item) {
            return sum + item.width * item.height;
        }
    );
    size_t min_height_estimation = ceil(full_area / (double)unit_width);

    size_t max_item_height = max_element(rectangles.begin(), rectangles.end(), [](const rectangle& item1, const rectangle& item2) {
            return item1.height < item2.height;
        }
    )->height;

    min_height_estimation = max(min_height_estimation, max_item_height);
    return min_height_estimation;
}


// в начальную строку подходит, поэтому смотрим следующие (i = i_min + 1)
bool is_fit(const size_t& i_min, const size_t& x_left, const rectangle& item, const vector<forward_list<interval>>& rows) {
    bool fit = true; // для item height == 1
    for (size_t i = i_min + 1; i < i_min + item.height; i++) {
        fit = false;
        for (const auto& local_interval : rows[i]) {
            if (local_interval.left > x_left)
                break;

            if (local_interval.left <= x_left && local_interval.right >= x_left + item.width) {
                fit = true;
                break;
            }
        }
        if (!fit)
            break;
    }
    return fit;
}


void update_rows(const size_t& i_min, const size_t& x_left, const rectangle& item, vector<forward_list<interval>>& rows) {
    size_t x_right = x_left + item.width;
    for (size_t i = i_min; i < i_min + item.height; i++) {
        forward_list<interval>& row = rows[i];

        auto prev_local_interval = row.before_begin();
        auto local_interval = row.begin();
        while (local_interval != row.end()) {
            if (local_interval->right > x_left) {
                assert(local_interval->left <= x_left);
                assert(local_interval->right >= x_right);
                
                if (local_interval->left < x_left) {
                    if (local_interval->right == x_right) {
                        local_interval->right = x_left;
                        break;
                    }
                    else {
                        row.emplace_after(local_interval, x_right, local_interval->right);
                        local_interval->right = x_left;
                        break;
                    }
                }
                else {
                    if (local_interval->right == x_right) {
                        row.erase_after(prev_local_interval);
                        break;
                    }
                    else {
                        local_interval->left = x_right;
                        break;
                    }
                }
            }
            prev_local_interval = local_interval;
            ++local_interval;
        }
    }
}


void recursion_core(const vector<rectangle>& rectangles, const size_t& unit_width,
    const size_t& k, const size_t& height, const vector<forward_list<interval>>& rows, vector<packed_rectangle>& packed_rectangles,
    const size_t& height_lower_bound, size_t& height_upper_bound, size_t& target_height, vector<packed_rectangle>& best_packed_rectangles) {

    const rectangle& item = rectangles[k];
    
    for (size_t i = 0; i <= target_height - item.height; i++) {
        const size_t new_height = max(height, i + item.height);
        for (const auto& current_interval : rows[i]) {
            if (current_interval.right - current_interval.left < item.width)
                continue;
            
            for (size_t x_left = current_interval.left; x_left <= current_interval.right - item.width; x_left++) {
                if (!is_fit(i, x_left, item, rows))
                    continue;

                bool is_last_item = k + 1 == rectangles.size();
                if (is_last_item) {
                    assert (new_height <= target_height);
                    
                    height_upper_bound = new_height;
                    target_height = height_upper_bound - 1;
                    best_packed_rectangles = packed_rectangles;
                    best_packed_rectangles.push_back({x_left, x_left + item.width, i, i + item.height});

                    cout << "update solution" << endl;
                    show_solution(unit_width, new_height, best_packed_rectangles);

                    return;
                }
                else {
                    vector<forward_list<interval>> new_rows{rows};
                    update_rows(i, x_left, item, new_rows);

                    packed_rectangles.push_back({x_left, x_left + item.width, i, i + item.height});

                    recursion_core(rectangles, unit_width,
                        k+1, new_height, new_rows, packed_rectangles,
                        height_lower_bound, height_upper_bound, target_height, best_packed_rectangles);
                    
                    packed_rectangles.pop_back();
                    
                    if (height_lower_bound >= height_upper_bound) {
                        cout << "lower_bound has been reached" << endl;
                        return;
                    }

                    if (new_height > target_height)
                        return;
                }
            }
        }
    }
}


// Decision tree and B&B
// packed_rectangles is valid packing with height = best_height
size_t exact_recursive(const size_t& unit_width, vector<rectangle>& rectangles, vector<packed_rectangle>& packed_rectangles, const size_t& best_height) {
    size_t height_lower_bound = min_height_estimation(unit_width, rectangles);
    size_t height_upper_bound = best_height;
    size_t target_height = height_upper_bound - 1;

    if (height_lower_bound >= height_upper_bound) {
        cout << "lower_bound has been reached" << endl;
        return height_upper_bound;
    }
    
    sort(rectangles.begin(), rectangles.end(), sort_by_area);

    vector<forward_list<interval>> rows(target_height);
    for (auto& row : rows) {
        auto iter = row.before_begin();
        row.emplace_after(iter, 0, unit_width);
    }
    
    vector<packed_rectangle> packing_sample;
    packing_sample.clear();
    packing_sample.reserve(packed_rectangles.size());

    recursion_core(rectangles, unit_width,
        0, 0, rows, packing_sample,
        height_lower_bound, height_upper_bound, target_height, packed_rectangles);
    
    return height_upper_bound;
}


// Decision tree and B&B
// packed_rectangles is valid packing with height = best_height
size_t exact(const size_t& unit_width, vector<rectangle>& rectangles, vector<packed_rectangle>& packed_rectangles, const size_t& best_height) {
    size_t height_lower_bound = min_height_estimation(unit_width, rectangles);
    size_t height_upper_bound = best_height;
    size_t target_height = height_upper_bound - 1;

    if (height_lower_bound >= height_upper_bound)
        return height_upper_bound;
    
    sort(rectangles.begin(), rectangles.end(), sort_by_area);

    vector<forward_list<interval>> rows(target_height);
    for (auto& row : rows) {
        auto iter = row.before_begin();
        row.emplace_after(iter, 0, unit_width);
    }
    
    vector<packed_rectangle> packing_sample;
    packing_sample.clear();
    packing_sample.reserve(packed_rectangles.size());
    stack<task> tasks;
    tasks.push({0, 0, rows, packing_sample});
    while (!tasks.empty()) {
        task current_task = tasks.top();
        tasks.pop();

        // target_height может изменится в другой ветке
        if (current_task.height > target_height)
            continue;

        const rectangle& item = rectangles[current_task.i];
        
        for (size_t i = 0; i <= target_height - item.height; i++) {
            for (const auto& current_interval : current_task.rows[i]) {
                if (current_interval.right - current_interval.left < item.width)
                    continue;
                
                for (size_t x_left = current_interval.left; x_left <= current_interval.right - item.width; x_left++) {
                    if (!is_fit(i, x_left, item, current_task.rows))
                        continue;

                    size_t height = max(current_task.height, i + item.height);
                    bool is_last_item = current_task.i + 1 == rectangles.size();
                    if (is_last_item) {
                        if (height <= target_height) {
                            height_upper_bound = height;
                            target_height = height_upper_bound - 1;
                            packed_rectangles = current_task.packed_items;
                            packed_rectangles.push_back({x_left, x_left + item.width, i, i + item.height});

                            if (height_lower_bound >= height_upper_bound) {
                                cout << "lower_bound achieved" << endl;
                                return height_upper_bound;
                            }
                                
                        }
                    }
                    else {
                        assert(height <= target_height);
                        task new_task{current_task};
                        new_task.i += 1;
                        new_task.height = height;
                        new_task.packed_items.push_back({x_left, x_left + item.width, i, i + item.height});

                        update_rows(i, x_left, item, new_task.rows);
                        tasks.push(new_task);
                    }
                }
            }
        }
    }
    return height_upper_bound;
}


size_t solve_problem_exact(const size_t& unit_width, const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights, vector<packed_rectangle>& packed_rectangles) {
    check_input(unit_width, items_number, widths, heights);

    vector<rectangle> rectangles{convert_input(items_number, widths, heights)};
    sort(rectangles.begin(), rectangles.end(), sort_by_height);
    
    vector<packed_rectangle> packed_rectangles_nfdh, packed_rectangles_skyline;
    size_t height_nfdh = nfdh(unit_width, rectangles, packed_rectangles_nfdh);
    size_t height_skyline = skyline(unit_width, rectangles, packed_rectangles_skyline);
    
    size_t best_height;
    if (height_nfdh <= height_skyline) {
        best_height = height_nfdh;
        packed_rectangles = packed_rectangles_nfdh;
    }
    else {
        best_height = height_skyline;
        packed_rectangles = packed_rectangles_skyline;
    }

    size_t height = exact_recursive(unit_width, rectangles, packed_rectangles, best_height);

    show_solution(unit_width, height, packed_rectangles);

    return height;
}
