"""
Improved waypoint scoring system with better local valley detection.

This module implements a more sophisticated scoring system based on proper
ant simulation and distance calculations as described in the research instructions.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Set, Optional
from collections import deque
import random


def simulate_ant_movement(screenshot: np.ndarray, waypoints: List[Dict[str, float]], 
                         ant_start: Tuple[int, int], max_steps: int = 1000) -> Tuple[bool, List[Tuple[int, int]]]:
    """
    Simulate an ant's movement following the distance heuristic.
    
    The ant tries to locally minimize its distance to the next waypoint,
    then to the sink. Returns whether the ant reaches a sink and the path taken.
    """
    height, width = screenshot.shape
    current_pos = ant_start
    path = [current_pos]
    current_waypoint_idx = 0
    
    # Find sink positions
    sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
    if not sinks:
        return False, path
    
    for step in range(max_steps):
        # Determine target: next waypoint or closest sink
        if current_waypoint_idx < len(waypoints):
            target_x = waypoints[current_waypoint_idx]['x']
            target_y = waypoints[current_waypoint_idx]['y']
            target_radius = waypoints[current_waypoint_idx]['radius']
            
            # Check if ant has reached current waypoint
            dist_to_waypoint = math.sqrt((current_pos[0] - target_x)**2 + (current_pos[1] - target_y)**2)
            if dist_to_waypoint <= target_radius:
                current_waypoint_idx += 1
                continue  # Move to next waypoint
        else:
            # Head to nearest sink
            target_x, target_y = min(sinks, key=lambda s: math.sqrt((current_pos[0] - s[0])**2 + (current_pos[1] - s[1])**2))
            
            # Check if ant has reached sink
            if (current_pos[0], current_pos[1]) in sinks:
                return True, path
        
        # Find best move (locally minimize distance to target)
        best_pos = None
        best_distance = float('inf')
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            new_x, new_y = current_pos[0] + dx, current_pos[1] + dy
            
            # Check bounds
            if new_x < 0 or new_x >= width or new_y < 0 or new_y >= height:
                continue
            
            # Check if passable (not wall)
            if screenshot[new_y, new_x] == 1:
                continue
            
            # Calculate distance to target
            distance = math.sqrt((new_x - target_x)**2 + (new_y - target_y)**2)
            
            if distance < best_distance:
                best_distance = distance
                best_pos = (new_x, new_y)
        
        if best_pos is None:
            # Ant is stuck (no valid moves)
            return False, path
        
        current_pos = best_pos
        path.append(current_pos)
        
        # Check if ant is repeating positions (stuck in loop)
        if len(path) > 20 and current_pos in path[-20:-1]:
            return False, path
    
    # Ant didn't reach sink within max_steps
    return False, path


def detect_local_valleys_proper(screenshot: np.ndarray, waypoints: List[Dict[str, float]], 
                               num_samples: int = 50) -> float:
    """
    Properly detect local valleys by simulating ant movement.
    
    Returns the fraction of ants that get stuck in local minima.
    """
    # NOTE: Empty waypoint lists should still be tested! 
    # Ants with no waypoints are MORE likely to get stuck, not less.
    
    # Find source positions
    sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
    if not sources:
        return 0.0
    
    stuck_count = 0
    total_simulations = 0
    
    # Sample different starting positions and add some noise
    height, width = screenshot.shape
    
    for _ in range(num_samples):
        # Pick a random source or nearby position
        if sources:
            base_source = random.choice(sources)
            # Add small random offset to test nearby positions
            for _ in range(3):  # Try a few positions near each source
                offset_x = random.randint(-5, 5)
                offset_y = random.randint(-5, 5)
                test_x = max(0, min(width - 1, base_source[0] + offset_x))
                test_y = max(0, min(height - 1, base_source[1] + offset_y))
                
                # Skip if position is a wall
                if screenshot[test_y, test_x] == 1:
                    continue
                
                total_simulations += 1
                reached_sink, path = simulate_ant_movement(screenshot, waypoints, (test_x, test_y))
                
                if not reached_sink:
                    stuck_count += 1
    
    if total_simulations == 0:
        return 0.0
    
    return stuck_count / total_simulations


def calculate_path_efficiency_accurate(screenshot: np.ndarray, waypoints: List[Dict[str, float]]) -> float:
    """
    Calculate path efficiency using actual ant simulation.
    
    Returns the ratio of actual path length to theoretical optimal path length.
    """
    sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
    sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
    
    if not sources or not sinks:
        return 1.0
    
    total_actual_length = 0.0
    total_optimal_length = 0.0
    successful_simulations = 0
    
    for source in sources[:min(5, len(sources))]:  # Limit to avoid long computation
        reached_sink, path = simulate_ant_movement(screenshot, waypoints, source)
        
        if reached_sink and len(path) > 1:
            # Calculate actual path length
            actual_length = sum(
                math.sqrt((path[i][0] - path[i-1][0])**2 + (path[i][1] - path[i-1][1])**2)
                for i in range(1, len(path))
            )
            
            # Calculate optimal direct distance to nearest sink
            nearest_sink = min(sinks, key=lambda s: math.sqrt((source[0] - s[0])**2 + (source[1] - s[1])**2))
            optimal_length = math.sqrt((source[0] - nearest_sink[0])**2 + (source[1] - nearest_sink[1])**2)
            
            if optimal_length > 0:
                total_actual_length += actual_length
                total_optimal_length += optimal_length
                successful_simulations += 1
    
    if successful_simulations == 0 or total_optimal_length == 0:
        return 1.0
    
    return total_actual_length / total_optimal_length


def score_waypoint_quality(screenshot: np.ndarray, waypoints: List[Dict[str, float]]) -> float:
    """
    Calculate a quality score based on how well ants perform with these waypoints.
    
    This simulates multiple ants and measures their success rate and efficiency.
    """
    sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
    if not sources:
        return 1000.0  # No sources, major problem
    
    success_count = 0
    total_attempts = min(20, len(sources) * 5)  # Limit computation
    
    for _ in range(total_attempts):
        source = random.choice(sources)
        reached_sink, path = simulate_ant_movement(screenshot, waypoints, source, max_steps=500)
        if reached_sink:
            success_count += 1
    
    success_rate = success_count / total_attempts if total_attempts > 0 else 0.0
    
    # Score based on success rate (higher success = lower score)
    if success_rate >= 0.9:
        return 0.0  # Excellent
    elif success_rate >= 0.7:
        return 10.0 * (0.9 - success_rate)  # Good
    elif success_rate >= 0.5:
        return 20.0 + 50.0 * (0.7 - success_rate)  # Mediocre
    else:
        return 100.0 + 200.0 * (0.5 - success_rate)  # Poor


def improved_score_waypoint_list(screenshot: np.ndarray, waypoints: List[Dict[str, float]], 
                                penalize_skippable: bool = True, 
                                feature_flags: Optional[Dict[str, bool]] = None) -> float:
    """
    Improved scoring function with better local valley detection and evidence-based metrics.
    
    Based on research into optimal waypoint placement and ant simulation.
    """
    if feature_flags is None:
        feature_flags = {
            'use_ant_simulation': True,
            'check_local_valleys_proper': True,
            'check_path_efficiency_accurate': True,
            'check_waypoint_density': True,
            'check_coverage_gaps': True,
        }
    
    score = 0.0
    
    # Import original non-skippability checker
    try:
        from .waypoint_scoring import check_waypoint_non_skippability
    except ImportError:
        from waypoint_scoring import check_waypoint_non_skippability
    
    # Core requirement: non-skippability
    is_non_skippable = check_waypoint_non_skippability(screenshot, waypoints)
    if penalize_skippable and not is_non_skippable:
        score += 10000.0  # Heavy penalty
    
    # Basic connectivity check
    sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
    sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
    if not sources or not sinks:
        score += 5000.0  # Level is unsolvable
        return score
    
    # Primary metric: ant simulation quality
    if feature_flags.get('use_ant_simulation', True):
        quality_score = score_waypoint_quality(screenshot, waypoints)
        score += quality_score * 1.0  # Primary component
    
    # Local valley detection (proper implementation)
    if feature_flags.get('check_local_valleys_proper', True):
        valley_fraction = detect_local_valleys_proper(screenshot, waypoints, num_samples=30)
        score += valley_fraction * 500.0  # Heavy penalty for valleys
    
    # Path efficiency (accurate calculation)
    if feature_flags.get('check_path_efficiency_accurate', True):
        efficiency_ratio = calculate_path_efficiency_accurate(screenshot, waypoints)
        if efficiency_ratio > 2.0:  # Path more than 2x longer than optimal
            score += (efficiency_ratio - 2.0) * 50.0
    
    # Waypoint density penalty (avoid too many waypoints in small areas)
    if feature_flags.get('check_waypoint_density', True):
        if len(waypoints) > 1:
            min_distance = float('inf')
            for i in range(len(waypoints)):
                for j in range(i + 1, len(waypoints)):
                    wp1, wp2 = waypoints[i], waypoints[j]
                    dist = math.sqrt((wp1['x'] - wp2['x'])**2 + (wp1['y'] - wp2['y'])**2)
                    min_distance = min(min_distance, dist)
            
            if min_distance < 10.0:  # Waypoints too close together
                score += (10.0 - min_distance) * 5.0
    
    # Coverage gap detection (areas where ants might get lost)
    if feature_flags.get('check_coverage_gaps', True):
        coverage_penalty = _check_coverage_gaps(screenshot, waypoints)
        score += coverage_penalty
    
    # Standard penalties (smaller weights now)
    score += len(waypoints) * 2.0  # Mild penalty for more waypoints
    
    return score


def _check_coverage_gaps(screenshot: np.ndarray, waypoints: List[Dict[str, float]]) -> float:
    """
    Check for areas where ants might get lost due to poor waypoint coverage.
    
    This looks for large passable areas that are far from any waypoint.
    """
    if len(waypoints) == 0:
        return 0.0
    
    height, width = screenshot.shape
    gap_penalty = 0.0
    sample_spacing = max(8, min(width, height) // 15)
    
    for y in range(0, height, sample_spacing):
        for x in range(0, width, sample_spacing):
            if screenshot[y, x] != 1:  # Not a wall
                # Find distance to nearest waypoint
                min_waypoint_dist = float('inf')
                for wp in waypoints:
                    dist = math.sqrt((x - wp['x'])**2 + (y - wp['y'])**2)
                    min_waypoint_dist = min(min_waypoint_dist, dist)
                
                # Penalty if too far from any waypoint
                if min_waypoint_dist > 40.0:
                    gap_penalty += (min_waypoint_dist - 40.0) * 0.5
    
    return gap_penalty


def benchmark_scoring_functions(screenshot: np.ndarray, waypoints_list: List[List[Dict[str, float]]]) -> Dict[str, List[float]]:
    """
    Benchmark different scoring functions to compare their effectiveness.
    
    Returns scores from different scoring methods for analysis.
    """
    results = {
        'original': [],
        'improved': [],
        'ant_simulation_only': [],
        'local_valleys_only': []
    }
    
    # Import original scoring function
    try:
        from .waypoint_scoring import score_waypoint_list
    except ImportError:
        from waypoint_scoring import score_waypoint_list
    
    for waypoints in waypoints_list:
        # Original scoring
        original_score = score_waypoint_list(screenshot, waypoints, penalize_skippable=True)
        results['original'].append(original_score)
        
        # Improved scoring
        improved_score = improved_score_waypoint_list(screenshot, waypoints)
        results['improved'].append(improved_score)
        
        # Ant simulation only
        ant_only_score = score_waypoint_quality(screenshot, waypoints)
        results['ant_simulation_only'].append(ant_only_score)
        
        # Local valleys only
        valley_score = detect_local_valleys_proper(screenshot, waypoints) * 500.0
        results['local_valleys_only'].append(valley_score)
    
    return results


if __name__ == "__main__":
    # Test the improved scoring system
    import sys
    import os
    
    # Add py_autotweaker to path
    sys.path.append(os.path.dirname(__file__))
    
    from waypoint_test_runner import create_synthetic_test_cases
    
    print("Testing improved waypoint scoring system...")
    
    # Create test cases
    test_cases = create_synthetic_test_cases()
    
    for test_name, screenshot in test_cases:
        print(f"\nTesting on {test_name}:")
        
        # Test with no waypoints
        empty_waypoints = []
        score = improved_score_waypoint_list(screenshot, empty_waypoints)
        print(f"  Empty waypoints score: {score:.2f}")
        
        # Test with a simple waypoint
        if test_name == "synthetic_l_shape":
            test_waypoints = [{'x': 30.0, 'y': 15.0, 'radius': 5.0}]
            score = improved_score_waypoint_list(screenshot, test_waypoints)
            print(f"  Single waypoint score: {score:.2f}")
    
    print("\nImproved scoring system test completed!")