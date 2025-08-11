#include <iostream>
#include <cstdint>
#include <vector>
#include <iomanip>
#include <limits>
#include "fcsim.h"

struct waypoint_t {
    double x;
    double y;
    double radius;
};

std::istream& operator>>(std::istream& is, fcsim_block_def& block) {
    int64_t block_type;
    is >> block_type >> block.id >> block.x >> block.y >> block.w >> block.h >> block.angle >> block.joints[0] >> block.joints[1];
    block.type = static_cast<fcsim_piece_type::type>(block_type);
    return is;
}

std::istream& operator>>(std::istream& is, fcsim_rect& rect) {
    return is >> rect.x >> rect.y >> rect.w >> rect.h;
}

std::istream& operator>>(std::istream& is, ft_design_spec& design) {
    int64_t num_blocks;
    is >> num_blocks;
    for(int64_t i = 0; i < num_blocks; ++i) {
        fcsim_block_def block;
        is >> block;
        design.blocks.push_back(block);
    }
    is >> design.build >> design.goal;
    return is;
}

std::istream& operator>>(std::istream& is, waypoint_t& waypoint) {
    return is >> waypoint.x >> waypoint.y >> waypoint.radius;
}

template <typename T>
std::istream& operator>>(std::istream& is, std::vector<T>& vec) {
    int64_t length;
    is >> length;
    std::vector<T> result;
    for(int64_t i = 0; i < length; ++i) {
        T item;
        is >> item;
        result.push_back(item);
    }
    return is;
}

// extremely low level (not for human use) program to run a single design, and see if it solves
int main() {
    // prepare double output to be max precision
    std::cout << std::setprecision(std::numeric_limits<double>::max_digits10);
    // read parameters from stdin
    int64_t max_ticks;
    std::cin >> max_ticks;
    // read design data from stdin
    ft_design_spec design;
    std::cin >> design;
    // additional parameters for autotweaker
    int64_t min_ticks;
    std::cin >> min_ticks;
    std::vector<waypoint_t> all_waypoints;
    std::cin >> all_waypoints;
    // placeholder
    double best_score = 1e300;
    // run up to max ticks
    std::shared_ptr<ft_sim_state> handle;
    ft_sim_settings settings;
    handle = fcsim_new(handle, design, settings);
    int64_t solve_tick = -1;
    while(handle->tick != max_ticks) {
        fcsim_step(handle, settings);
        if(fcsim_is_solved(handle, design)) {
            // set placeholder
            best_score = 0;
            // stop early on solve
            solve_tick = handle->tick;
            break;
        }
    }
    // print result
    std::cout << solve_tick << std::endl << handle->tick << std::endl << best_score << std::endl;
    return 0;
}