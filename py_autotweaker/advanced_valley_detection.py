"""
Advanced local valley detection using numpy optimizations and mathematical analysis.

This module provides highly granular local valley detection that can identify
1-pixel corners and subtle local minima using efficient numpy operations
and topological analysis rather than just random sampling.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Set
from scipy import ndimage
from collections import deque
import random

try:
    from .improved_waypoint_scoring import simulate_ant_movement
except ImportError:
    from improved_waypoint_scoring import simulate_ant_movement


def compute_distance_field(screenshot: np.ndarray, waypoints: List[Dict[str, float]]) -> np.ndarray:
    """
    Compute the distance field that ants follow - distance to next waypoint or sink.
    
    This creates a 2D array where each pixel contains the distance an ant at that
    position would need to travel to reach its target (next waypoint or sink).
    """
    height, width = screenshot.shape
    
    # Find sinks
    sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
    if not sinks:
        return np.full((height, width), float('inf'))
    
    if len(waypoints) == 0:
        # No waypoints - distance to nearest sink
        distance_field = np.full((height, width), float('inf'))
        
        for y in range(height):
            for x in range(width):
                if screenshot[y, x] != 1:  # Not a wall
                    min_dist = min(math.sqrt((x - sx)**2 + (y - sy)**2) for sx, sy in sinks)
                    distance_field[y, x] = min_dist
        
        return distance_field
    
    else:
        # With waypoints - more complex distance calculation
        # For now, simplified to first waypoint distance
        first_waypoint = waypoints[0]
        wx, wy, wr = first_waypoint['x'], first_waypoint['y'], first_waypoint['radius']
        
        distance_field = np.full((height, width), float('inf'))
        
        for y in range(height):
            for x in range(width):
                if screenshot[y, x] != 1:  # Not a wall
                    # Distance to waypoint edge (not center)
                    dist_to_center = math.sqrt((x - wx)**2 + (y - wy)**2)
                    dist_to_waypoint = max(0, dist_to_center - wr)
                    distance_field[y, x] = dist_to_waypoint
        
        return distance_field


def detect_local_minima_numpy(distance_field: np.ndarray, screenshot: np.ndarray) -> np.ndarray:
    """
    Efficiently detect local minima in the distance field using numpy operations.
    
    A local minimum is a position where all neighboring positions have higher
    or equal distance values - these are potential ant traps.
    """
    height, width = distance_field.shape
    
    # Create mask for passable areas (not walls)
    passable_mask = (screenshot != 1)
    
    # Use scipy's ndimage to efficiently find local minima
    # A point is a local minimum if it's <= all neighbors
    local_min_mask = ndimage.minimum_filter(distance_field, size=3) == distance_field
    
    # Only consider passable areas
    local_min_mask = local_min_mask & passable_mask
    
    # Exclude trivial minima (sinks and waypoint centers)
    # Sinks should be global minima, not problematic local minima
    sinks = (screenshot == 4)
    local_min_mask = local_min_mask & ~sinks
    
    return local_min_mask


def detect_gradient_traps_numpy(distance_field: np.ndarray, screenshot: np.ndarray) -> np.ndarray:
    """
    Detect gradient traps - areas where the gradient points toward a local minimum.
    
    Uses numpy gradient computation for efficiency.
    """
    height, width = distance_field.shape
    
    # Compute gradients using numpy (handle infinite values)
    finite_distance_field = np.where(np.isfinite(distance_field), distance_field, 0)
    grad_y, grad_x = np.gradient(finite_distance_field)
    
    # Gradient magnitude
    grad_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Areas with very low gradient are potential traps (flat regions)
    low_gradient_mask = grad_magnitude < 0.1
    
    # Only consider passable areas
    passable_mask = (screenshot != 1)
    low_gradient_mask = low_gradient_mask & passable_mask
    
    # Exclude sinks (they should have low gradient)
    sinks = (screenshot == 4)
    low_gradient_mask = low_gradient_mask & ~sinks
    
    return low_gradient_mask


def find_corner_traps_numpy(screenshot: np.ndarray) -> np.ndarray:
    """
    Efficiently find corner configurations that could trap ants.
    
    Uses convolution-like operations to detect corner patterns.
    """
    height, width = screenshot.shape
    
    # Define corner detection kernels
    corner_kernels = [
        # L-shaped corners (4 orientations)
        np.array([[1, 1, 0],
                  [1, 0, 0], 
                  [0, 0, 0]]),
        np.array([[0, 1, 1],
                  [0, 0, 1],
                  [0, 0, 0]]),
        np.array([[0, 0, 0],
                  [0, 0, 1],
                  [0, 1, 1]]),
        np.array([[0, 0, 0],
                  [1, 0, 0],
                  [1, 1, 0]]),
        
        # Narrow passages
        np.array([[1, 0, 1],
                  [1, 0, 1],
                  [1, 0, 1]]),
        np.array([[1, 1, 1],
                  [0, 0, 0],
                  [1, 1, 1]]),
    ]
    
    corner_mask = np.zeros((height, width), dtype=bool)
    
    # Apply each kernel using correlation
    for kernel in corner_kernels:
        # Correlate with wall pattern (1 = wall, 0 = passable)
        convolved = ndimage.correlate(screenshot.astype(np.float32), kernel.astype(np.float32), mode='constant')
        
        # Locations where kernel pattern matches exactly
        kernel_sum = np.sum(kernel)
        matches = (convolved == kernel_sum)
        
        # The center pixel should be passable for it to be a trap
        passable_center = (screenshot != 1)
        corner_mask |= (matches & passable_center)
    
    return corner_mask


def systematic_valley_detection(screenshot: np.ndarray, waypoints: List[Dict[str, float]], 
                               granularity: int = 1) -> Tuple[float, Dict[str, any]]:
    """
    Systematic local valley detection using multiple mathematical approaches.
    
    Args:
        screenshot: Level screenshot
        waypoints: List of waypoints
        granularity: Pixel sampling granularity (1 = every pixel, 2 = every other pixel, etc.)
    
    Returns:
        Tuple of (valley_fraction, detailed_analysis)
    """
    height, width = screenshot.shape
    
    # Step 1: Compute distance field
    distance_field = compute_distance_field(screenshot, waypoints)
    
    # Step 2: Find different types of problematic areas
    local_minima = detect_local_minima_numpy(distance_field, screenshot)
    gradient_traps = detect_gradient_traps_numpy(distance_field, screenshot)
    corner_traps = find_corner_traps_numpy(screenshot)
    
    # Combine all trap types
    all_traps = local_minima | gradient_traps | corner_traps
    
    # Step 3: Systematic sampling for validation
    # Sample positions throughout the level, not just near sources
    passable_positions = [(x, y) for y in range(0, height, granularity) 
                         for x in range(0, width, granularity)
                         if screenshot[y, x] != 1]
    
    if len(passable_positions) == 0:
        return 0.0, {'error': 'No passable positions found'}
    
    # Step 4: Intelligent sampling - prioritize trap areas and random sampling
    trap_positions = [(x, y) for y, x in zip(*np.where(all_traps))]
    
    # Sample more densely in trap areas, plus random sampling
    test_positions = []
    
    # Add trap positions (up to 50% of samples)
    max_trap_samples = min(len(trap_positions), len(passable_positions) // 2)
    if trap_positions and max_trap_samples > 0:
        test_positions.extend(random.sample(trap_positions, max_trap_samples))
    
    # Add random positions for general coverage
    remaining_samples = max(50, len(passable_positions) // 10)  # At least 50 samples
    random_positions = random.sample(passable_positions, 
                                   min(remaining_samples, len(passable_positions)))
    test_positions.extend(random_positions)
    
    # Remove duplicates
    test_positions = list(set(test_positions))
    
    # Step 5: Test ant movement from selected positions
    stuck_count = 0
    total_tests = len(test_positions)
    
    position_results = []
    
    for x, y in test_positions:
        reached_sink, path = simulate_ant_movement(screenshot, waypoints, (x, y))
        
        position_results.append({
            'position': (x, y),
            'reached_sink': reached_sink,
            'path_length': len(path) if path else 0,
            'in_trap_area': all_traps[y, x] if 0 <= y < height and 0 <= x < width else False
        })
        
        if not reached_sink:
            stuck_count += 1
    
    valley_fraction = stuck_count / total_tests if total_tests > 0 else 0.0
    
    # Detailed analysis
    analysis = {
        'valley_fraction': valley_fraction,
        'total_tests': total_tests,
        'stuck_count': stuck_count,
        'trap_analysis': {
            'local_minima_count': int(np.sum(local_minima)),
            'gradient_trap_count': int(np.sum(gradient_traps)), 
            'corner_trap_count': int(np.sum(corner_traps)),
            'total_trap_pixels': int(np.sum(all_traps))
        },
        'sampling_info': {
            'granularity': granularity,
            'total_passable_pixels': len(passable_positions),
            'trap_positions_tested': max_trap_samples,
            'random_positions_tested': len(random_positions)
        },
        'position_results': position_results[:10]  # First 10 for debugging
    }
    
    return valley_fraction, analysis


def quick_valley_detection_numpy(screenshot: np.ndarray, waypoints: List[Dict[str, float]]) -> float:
    """
    Fast valley detection using pure numpy operations for performance.
    
    This version prioritizes speed over detailed analysis.
    """
    height, width = screenshot.shape
    
    # Fast distance field computation
    sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
    if not sinks:
        return 0.0
    
    # Create coordinate grids
    y_coords, x_coords = np.ogrid[:height, :width]
    
    if len(waypoints) == 0:
        # Distance to nearest sink using broadcasting
        sink_distances = []
        for sx, sy in sinks:
            dist = np.sqrt((x_coords - sx)**2 + (y_coords - sy)**2)
            sink_distances.append(dist)
        
        distance_field = np.minimum.reduce(sink_distances)
    else:
        # Distance to first waypoint
        wx, wy, wr = waypoints[0]['x'], waypoints[0]['y'], waypoints[0]['radius']
        distance_field = np.maximum(0, np.sqrt((x_coords - wx)**2 + (y_coords - wy)**2) - wr)
    
    # Set walls to infinite distance
    distance_field[screenshot == 1] = np.inf
    
    # Fast local minima detection
    local_min_mask = ndimage.minimum_filter(distance_field, size=3) == distance_field
    passable_mask = (screenshot != 1) & (screenshot != 4)  # Exclude walls and sinks
    problematic_areas = local_min_mask & passable_mask
    
    # Estimate valley fraction based on problematic area ratio
    total_passable = np.sum(passable_mask)
    problematic_pixels = np.sum(problematic_areas)
    
    if total_passable == 0:
        return 0.0
    
    # Convert spatial analysis to movement probability
    # More problematic pixels = higher chance ants get stuck
    spatial_ratio = problematic_pixels / total_passable
    
    # Apply a scaling factor based on empirical testing
    # This converts spatial analysis to movement simulation equivalent
    valley_fraction = min(1.0, spatial_ratio * 5.0)  # Scale factor may need tuning
    
    return valley_fraction


if __name__ == "__main__":
    # Test the new valley detection methods
    print("Testing advanced valley detection...")
    
    import sys
    sys.path.append('.')
    
    from waypoint_test_runner import create_synthetic_test_cases
    
    test_cases = create_synthetic_test_cases()
    
    for test_name, screenshot in test_cases:
        print(f"\nTesting {test_name}:")
        
        # Test with no waypoints (Null algorithm)
        null_waypoints = []
        
        # Original method for comparison
        from improved_waypoint_scoring import detect_local_valleys_proper
        original_fraction = detect_local_valleys_proper(screenshot, null_waypoints, num_samples=30)
        
        # New systematic method
        new_fraction, analysis = systematic_valley_detection(screenshot, null_waypoints, granularity=2)
        
        # Quick numpy method
        quick_fraction = quick_valley_detection_numpy(screenshot, null_waypoints)
        
        print(f"  Original method: {original_fraction:.3f}")
        print(f"  Systematic method: {new_fraction:.3f}")
        print(f"  Quick numpy method: {quick_fraction:.3f}")
        print(f"  Trap analysis: {analysis['trap_analysis']}")
        print(f"  Tests performed: {analysis['total_tests']}")
    
    print("\nAdvanced valley detection testing completed!")