#include <algorithm>
#include <cmath>
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
    vec.clear();
    for(int64_t i = 0; i < length; ++i) {
        T item;
        is >> item;
        vec.push_back(item);
    }
    return is;
}

static bool is_goal_object(fcsim_piece_type::type type)
{
	switch (type) {
	case fcsim_piece_type::GP_RECT:
	case fcsim_piece_type::GP_CIRC:
		return true;
	}
	return false;
}

double update_score(std::shared_ptr<ft_sim_state> handle, const ft_design_spec& design,
    const std::vector<waypoint_t>& all_waypoints, std::vector<int64_t>& waypoints_progress) {
    int64_t num_waypoints = (int64_t)all_waypoints.size();
    double total_score = 0;
    int gp_index = 0;
    for(int i = 0; i < handle->block_cnt; ++i) {
		fcsim_block_def& bdef = handle->blocks[i].bdef;
		if(!is_goal_object(bdef.type))continue;
        double gp_score_part = 0;
        bool gp_is_in_goal = false;
        // make sure waypoints_progress is long enough
		while(gp_index >= waypoints_progress.size()) {
            waypoints_progress.emplace_back(0);
        }
        // consume waypoints if possible
        while(waypoints_progress[gp_index] < num_waypoints &&
            hypot(bdef.x - all_waypoints[waypoints_progress[gp_index]].x,
                bdef.y - all_waypoints[waypoints_progress[gp_index]].y) <
                all_waypoints[waypoints_progress[gp_index]].radius) {
            waypoints_progress[gp_index]++;
        }
        // check our progress
        int64_t progress_ticker = waypoints_progress[gp_index];
        double x = bdef.x;
        double y = bdef.y;
        if(progress_ticker < num_waypoints) {
            // we still have more waypoints to consume
            while(progress_ticker < num_waypoints) {
                gp_score_part += hypot(x - all_waypoints[progress_ticker].x, y - all_waypoints[progress_ticker].y);
                x = all_waypoints[progress_ticker].x;
                y = all_waypoints[progress_ticker].y;
                progress_ticker++;
            }
            gp_score_part += hypot(x - design.goal.x, y - design.goal.y);
        } else {
            // only the goal remains
            // are we in the goal?
            double w = bdef.w;
            double h = bdef.h;
            double dx;
            double dy;
            if(bdef.type == fcsim_piece_type::GP_RECT) {
                double c = std::abs(ft_cos(bdef.angle));
                double s = std::abs(ft_sin(bdef.angle));
                dx = c * w + s * h;
                dy = s * w + c * h;
            } else {
                dx = dy = w;
            }
            dx *= 0.5;
            dy *= 0.5;
            double x1 = x - dx;
            double x2 = x + dx;
            double y1 = y - dy;
            double y2 = y + dy;
            double gx1 = design.goal.x - design.goal.w * 0.5;
            double gx2 = design.goal.x + design.goal.w * 0.5;
            double gy1 = design.goal.y - design.goal.h * 0.5;
            double gy2 = design.goal.y + design.goal.h * 0.5;
            if(gx1 <= x1 && x2 <= gx2 && gy1 <= y1 && y2 <= gy2) {
                // seems like it's in the goal area
                gp_is_in_goal = true;
                // add safety margin bonus
                gp_score_part -= hypot(
                    std::min(x1 - gx1, gx2 - x2),
                    std::min(y1 - gy1, gy2 - y2)
                );
            } else {
                // not in the goal
                // add distance to goal center
                gp_score_part += hypot(x - design.goal.x, y - design.goal.y);
            }
        }
        // apply outside of goal penalty
        gp_score_part += (gp_is_in_goal ? 0 : 10000);
        total_score += gp_score_part;
        gp_index++;
	}
    return total_score;
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
    // track best score
    double best_score = 1e300;
    std::vector<int64_t> waypoints_progress;
    // run up to max ticks
    std::shared_ptr<ft_sim_state> handle;
    ft_sim_settings settings;
    handle = fcsim_new(handle, design, settings);
    int64_t solve_tick = -1;
    while(handle->tick != max_ticks) {
        fcsim_step(handle, settings);
        best_score = std::min(best_score, update_score(handle, design, all_waypoints, waypoints_progress));
        if(solve_tick == -1 && fcsim_is_solved(handle, design)) {
            // record first time solved
            solve_tick = handle->tick;
        }
    }
    // print result
    std::cout << solve_tick << std::endl << handle->tick << std::endl << best_score << std::endl;
    return 0;
}