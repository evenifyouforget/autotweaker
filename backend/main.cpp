#include <iostream>
#include <cstdint>
#include "fcsim.h"

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

// extremely low level (not for human use) program to run a single design, and see if it solves
int main() {
    // read parameters from stdin
    int64_t max_ticks;
    std::cin >> max_ticks;
    // read design data from stdin
    ft_design_spec design;
    std::cin >> design;
    // run up to max ticks
    std::shared_ptr<ft_sim_state> handle;
    ft_sim_settings settings;
    handle = fcsim_new(handle, design, settings);
    int64_t solve_tick = -1;
    while(handle->tick != max_ticks) {
        fcsim_step(handle, settings);
        if(fcsim_is_solved(handle, design)) {
            // stop early on solve
            solve_tick = handle->tick;
            break;
        }
    }
    // print result
    std::cout << solve_tick << std::endl << handle->tick << std::endl;
    return 0;
}