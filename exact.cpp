#include <iostream>
#include <vector>
#include <utility>
#include <algorithm>
#include <cassert>
#include <stack>
#include <forward_list>
#include <numeric>
#include <cmath>
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


void check_input(const size_t& unit_width, const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights) {
    assert(widths.size() == heights.size() && items_number == widths.size());

    for (auto& width : widths)
        assert(width <= unit_width);
}


bool sort_by_height(const rectangle& a, const rectangle& b) {
    return a.height > b.height;
}


bool sort_by_area(const rectangle& a, const rectangle& b) {
    return a.width * a.height > b.width * b.height;
}


void show_solution(const size_t& unit_width, const size_t& max_height, const vector<packed_rectangle>& packed_rectangles) {
    vector<string> image(max_height);
    for (auto& row : image) {
        row.reserve(unit_width);
        for (size_t j = 0; j < unit_width; j++)
            row.push_back(' ');
    }
    
    string symbols{"!\"#$%&'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}~"};
    //assert(packed_rectangles.size() <= symbols.size());
    
    for (size_t i = 0; i < packed_rectangles.size(); i++) {
        char symbol = symbols[i % symbols.size()];
        for (size_t j = packed_rectangles[i].y_min; j < packed_rectangles[i].y_max; j++)
        {
            assert(packed_rectangles[i].x_max <= unit_width);
            for (size_t k = packed_rectangles[i].x_min; k < packed_rectangles[i].x_max; k++)
            {
                assert(image[j][k] == ' ');
                image[j][k] = symbol;
            }
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
size_t nfdh(const size_t& unit_width, const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights, vector<packed_rectangle>& packed_rectangles) {
    check_input(unit_width, items_number, widths, heights);

    vector<rectangle> rectangles;
    rectangles.reserve(items_number);
    for (size_t i = 0; i < items_number; i++)
        rectangles.push_back({widths[i], heights[i]});
    
    sort(rectangles.begin(), rectangles.end(), sort_by_height);

    packed_rectangles.clear();
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

    return shelf_max_height;
}


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
size_t skyline(const size_t& unit_width, const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights, vector<packed_rectangle>& packed_rectangles) {
    check_input(unit_width, items_number, widths, heights);

    vector<rectangle> rectangles;
    rectangles.reserve(items_number);
    for (size_t i = 0; i < items_number; i++)
        rectangles.push_back({widths[i], heights[i]});
    
    sort(rectangles.begin(), rectangles.end(), sort_by_height);

    packed_rectangles.clear();
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

    return max_height;
}

struct task {
    size_t i, height;
    vector<forward_list<interval>> rows;
    vector<packed_rectangle> packed_items;
};


size_t exact(const size_t& unit_width, const size_t& items_number, const vector<size_t>& widths, const vector<size_t>& heights, vector<packed_rectangle>& packed_rectangles) {
    vector<packed_rectangle> packed_rectangles_nfdh, packed_rectangles_skyline;
    size_t height_nfdh = nfdh(unit_width, items_number, widths, heights, packed_rectangles_nfdh);
    size_t height_skyline = skyline(unit_width, items_number, widths, heights, packed_rectangles_skyline);

    vector<rectangle> rectangles;
    rectangles.reserve(items_number);
    for (size_t i = 0; i < items_number; i++)
        rectangles.push_back({widths[i], heights[i]});
    
    size_t best_height;
    if (height_nfdh <= height_skyline) {
        best_height = height_nfdh;
        packed_rectangles = packed_rectangles_nfdh;
    }
    else {
        best_height = height_skyline;
        packed_rectangles = packed_rectangles_skyline;
    }
    size_t target_height = best_height - 1;
    
    // оценка снизу
    size_t possible_best_height = accumulate(
        rectangles.begin(),
        rectangles.end(), 
        0,
        [](size_t sum, const rectangle& item) {
            return sum + item.width * item.height;
        }
    );
    cout << possible_best_height << endl;
    possible_best_height = ceil(possible_best_height / (double)unit_width);
    cout << possible_best_height << endl;
    // добавить еще макс высоту прямоугольника

    if (possible_best_height >= best_height)
        return best_height;
    
    sort(rectangles.begin(), rectangles.end(), sort_by_area);

    vector<forward_list<interval>> rows(target_height);
    for (auto& row : rows) {
        auto iter = row.before_begin();
        row.emplace_after(iter, 0, unit_width);
    }
        //row.push_back({0, unit_width});
    
    vector<packed_rectangle> packing_sample;
    packing_sample.clear();
    packing_sample.reserve(items_number);
    stack<task> tasks;
    tasks.push({0, 0, rows, packing_sample});
    while (!tasks.empty()) {
        task current_task = tasks.top();
        tasks.pop();

        //show_solution(unit_width, best_height, packed_rectangles);

        // target_height может изменится (в другой ветке). Стоит смотреть только пока высота меньше.
        if (current_task.height > target_height)
            continue;

        rectangle item = rectangles[current_task.i];
        
        for (size_t i = 0; i <= target_height - item.height; i++)
        {
            // for (size_t j = 0; j < current_task.rows[i].size(); j++)
            for (const auto& current_interval : current_task.rows[i])
            {
                //interval current_interval = current_task.rows[i][j];

                if (current_interval.right - current_interval.left < item.width)
                    continue;
                
                for (size_t x_left = current_interval.left; x_left <= current_interval.right - item.width; x_left++)
                {
                    //cout << x_left << ", " << i << endl;
                    bool is_fit;
                    for (size_t il = i + 1; il < i + item.height; il++)
                    {
                        is_fit = false;
                        for (const auto& local_interval : current_task.rows[il])
                        {
                            if (local_interval.left > x_left)
                                break;

                            if (local_interval.right >= x_left + item.width && local_interval.left <= x_left) {
                                is_fit = true;
                                break;
                            }
                        }
                        if (!is_fit)
                            break;
                    }

                    if (!is_fit)
                        continue;
                    
                    cout << "Current iteration solution" << endl;
                    show_solution(unit_width, current_task.height, current_task.packed_items);
                    cout << "item to pack " << x_left << ", " << i << " " << item.width << ", " << item.height << endl;

                    for (size_t il = i; il < i + item.height; il++) {
                        cout << "i " << il << endl;
                        for (const auto& local_interval : current_task.rows[il])
                        {
                            cout << local_interval.left << " " << local_interval.right << endl; 
                        }
                    }
                        
                    
                    cout << "This was limits" << endl;

                    // нельзя изменять текущий таск, т.к. он используется в следующих итерациях
                    size_t height = max(current_task.height, i + item.height);
                    //cout << height << endl;
                    bool is_last_item = current_task.i + 1 == rectangles.size();
                    if (is_last_item) {
                        if (height <= target_height) {
                            best_height = height;
                            target_height = best_height - 1;
                            packed_rectangles = current_task.packed_items;
                            packed_rectangles.push_back({x_left, x_left + item.width, i, i + item.height});

                            if (possible_best_height >= best_height)
                                return best_height;
                        }
                    }
                    else {
                        assert(height <= target_height);
                        // не последняя, все правим и в стек кладем.
                        task new_task{current_task};
                        new_task.i += 1;
                        new_task.height = height;
                        
                        size_t x_right = x_left + item.width;
                        new_task.packed_items.push_back({x_left, x_right, i, i + item.height});

                        //cout << "here " << x_left << " " << x_right << endl;

                        // update rows
                        // но мы же знаем курент интервал, и даже остальные, которые в цикле чекали - сохранить в масиве
                        // но чтобы удалить надо знать предыдущий -> легче будет с обычным листом (а тут ща поиск)
                        for (size_t il = i; il < i + item.height; il++)
                        {
                            //cout << "row (i) " << il << endl;
                            forward_list<interval>& list = new_task.rows[il];

                            auto prev_local_interval = list.before_begin();
                            auto local_interval = list.begin();
                            while (local_interval != list.end()) {
                                //cout << "here" << endl;
                                //cout << "local_interval " << local_interval->left <<" " << local_interval->right << endl;
                                //if (local_interval->left >= x_right) // >
                                // if (local_interval->left > x_left) // оно будет даже больше х_правый
                                //     break;
                                if (local_interval->right > x_left) {
                                    assert(local_interval->left <= x_left);
                                    assert(local_interval->right >= x_right);
                                    
                                    if (local_interval->left < x_left) {
                                        if (local_interval->right == x_right) {
                                            local_interval->right = x_left;
                                            break;
                                        }
                                        else {
                                            //assert(local_interval->right >= x_right);
                                            list.emplace_after(local_interval, x_right, local_interval->right);
                                            local_interval->right = x_left;
                                            break;
                                        }
                                    }
                                    // local_interval->left == x_left
                                    else {
                                        if (local_interval->right == x_right) {
                                            //cout << "equal" << endl;
                                            //new_task.rows[il].erase_after(prev_local_interval);
                                            list.erase_after(prev_local_interval);
                                            //cout << "removed" << endl;
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
                            //cout << "here3" << endl;
                            //new_task.rows[il] = list;
                        }
                        cout << "Resulting solution" << endl;
                        show_solution(unit_width, new_task.height, new_task.packed_items);

                        for (size_t il = i; il < i + item.height; il++) {
                            cout << "i " << il << endl;
                            for (const auto& local_interval : new_task.rows[il])
                            {
                                cout << local_interval.left << " " << local_interval.right << endl; 
                            }
                        }
                        cout << "This was new limits" << endl;
                        //cout << "here2" << endl;
                        tasks.push(new_task);
                        //show_solution(unit_width, new_task.height, new_task.packed_items);
                    }
                }
            }
        }
    }

    show_solution(unit_width, best_height, packed_rectangles);

    return best_height;
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


void initialize_vectors_randomly2(const size_t& unit_width, const size_t& items_number, vector<size_t>& widths, vector<size_t>& heights) {
    widths.reserve(items_number);
    heights.reserve(items_number);
    for (size_t i = 0; i < items_number; i++)
    {
        widths.push_back(rand() % (unit_width / 2) + 1);
        heights.push_back(rand() % (unit_width / 2) + 1);
    }
}


void initialize_1(size_t& unit_width, size_t& items_number, vector<size_t>& widths, vector<size_t>& heights) {
    unit_width = 10;
    items_number = 6;
    widths = {2, 6, 4, 3, 5, 8};
    heights = {2, 4, 4, 3, 3, 2};
}


void initialize_3(size_t& unit_width, size_t& items_number, vector<size_t>& widths, vector<size_t>& heights) {
    unit_width = 10;
    items_number = 5;
    widths = {6, 4, 5, 5, 2};
    heights = {4, 4, 3, 3, 2};
}


void initialize_2(size_t& unit_width, size_t& items_number, vector<size_t>& widths, vector<size_t>& heights) {
    unit_width = 10;
    items_number = 10;

    initialize_vectors_randomly2(unit_width, items_number, widths, heights);
}


int main() {
    size_t unit_width;
    size_t items_number;
    vector<size_t> widths;
    vector<size_t> heights;

    initialize_2(unit_width, items_number, widths, heights);

    vector<packed_rectangle> packed_rectangles;
    size_t max_height = exact(unit_width, items_number, widths, heights, packed_rectangles);

    return 0;
}
