"""
Level validation utilities for waypoint generation testing.

Ensures levels meet basic requirements before testing waypoint algorithms.
As suggested in research instructions: "sanity check each level that a sink 
is actually reachable from every source, and if not, exclude it from the tournament."
"""

import numpy as np
from typing import List, Tuple, Dict, Set, Optional
from collections import deque

def is_level_valid(screenshot: np.ndarray, verbose: bool = False) -> Tuple[bool, str]:
    """
    Validate that a level meets basic requirements for waypoint testing.
    
    Args:
        screenshot: Level screenshot array
        verbose: Print detailed validation info
        
    Returns:
        Tuple of (is_valid, reason)
    """
    height, width = screenshot.shape
    
    # Find sources and sinks
    sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
    sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
    
    if len(sources) == 0:
        return False, "No sources found in level"
    
    if len(sinks) == 0:
        return False, "No sinks found in level"
    
    if verbose:
        print(f"Level validation: {len(sources)} sources, {len(sinks)} sinks")
    
    # Check if every source can reach at least one sink
    reachable_sources = 0
    unreachable_sources = []
    
    for i, source in enumerate(sources):
        can_reach_sink = False
        
        for sink in sinks:
            if is_path_exists(screenshot, source, sink):
                can_reach_sink = True
                break
        
        if can_reach_sink:
            reachable_sources += 1
        else:
            unreachable_sources.append(source)
    
    if unreachable_sources:
        reason = f"{len(unreachable_sources)}/{len(sources)} sources cannot reach any sink"
        if verbose:
            print(f"Unreachable sources: {unreachable_sources}")
        return False, reason
    
    # Additional validation checks
    passable_pixels = np.sum(screenshot != 1)
    total_pixels = height * width
    passable_ratio = passable_pixels / total_pixels
    
    if passable_ratio < 0.1:
        return False, f"Level too dense: only {passable_ratio:.1%} passable area"
    
    if passable_ratio > 0.95:
        return False, f"Level too sparse: {passable_ratio:.1%} passable area (walls needed)"
    
    if verbose:
        print(f"✅ Level valid: {reachable_sources}/{len(sources)} sources reachable, {passable_ratio:.1%} passable")
    
    return True, "Level passes all validation checks"


def is_path_exists(screenshot: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
    """
    Check if a path exists between start and end positions using BFS.
    
    Uses 4-directional movement (no diagonal) as specified in research instructions.
    """
    height, width = screenshot.shape
    
    if start == end:
        return True
    
    # BFS to find path
    queue = deque([start])
    visited = {start}
    
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # 4-directional
    
    while queue:
        x, y = queue.popleft()
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            
            # Check if already visited
            if (nx, ny) in visited:
                continue
            
            # Check if passable (not wall)
            if screenshot[ny, nx] == 1:
                continue
            
            # Found the end
            if (nx, ny) == end:
                return True
            
            # Add to queue
            queue.append((nx, ny))
            visited.add((nx, ny))
    
    return False


def validate_level_database(level_data: List[Tuple[str, np.ndarray]], 
                          verbose: bool = False) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, str]]]:
    """
    Validate a database of levels, returning valid levels and invalid reasons.
    
    Args:
        level_data: List of (level_name, screenshot) tuples
        verbose: Print validation details
        
    Returns:
        Tuple of (valid_levels, invalid_levels_with_reasons)
    """
    valid_levels = []
    invalid_levels = []
    
    if verbose:
        print(f"Validating {len(level_data)} levels...")
    
    for level_name, screenshot in level_data:
        is_valid, reason = is_level_valid(screenshot, verbose=False)
        
        if is_valid:
            valid_levels.append((level_name, screenshot))
            if verbose:
                print(f"✅ {level_name}: Valid")
        else:
            invalid_levels.append((level_name, reason))
            if verbose:
                print(f"❌ {level_name}: {reason}")
    
    if verbose:
        print(f"\nValidation complete: {len(valid_levels)}/{len(level_data)} levels valid")
    
    return valid_levels, invalid_levels


def analyze_level_characteristics(screenshot: np.ndarray) -> Dict[str, float]:
    """
    Analyze level characteristics for research insights.
    
    Returns dictionary of level metrics.
    """
    height, width = screenshot.shape
    
    # Basic counts
    walls = np.sum(screenshot == 1)
    sources = np.sum(screenshot == 3)
    sinks = np.sum(screenshot == 4)
    passable = np.sum((screenshot != 1))
    
    # Ratios
    wall_ratio = walls / (height * width)
    passable_ratio = passable / (height * width)
    
    # Connectivity analysis
    source_positions = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
    sink_positions = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
    
    # Calculate level complexity (rough estimate)
    complexity_score = 0
    if source_positions and sink_positions:
        # Distance from sources to sinks
        min_distances = []
        for sx, sy in source_positions:
            min_dist = min(abs(sx - tx) + abs(sy - ty) for tx, ty in sink_positions)
            min_distances.append(min_dist)
        
        avg_source_sink_distance = np.mean(min_distances) if min_distances else 0
        
        # Complexity based on wall density and path length
        complexity_score = wall_ratio * avg_source_sink_distance
    
    return {
        'width': width,
        'height': height, 
        'total_pixels': height * width,
        'walls': walls,
        'sources': sources,
        'sinks': sinks,
        'passable': passable,
        'wall_ratio': wall_ratio,
        'passable_ratio': passable_ratio,
        'avg_source_sink_distance': avg_source_sink_distance if 'avg_source_sink_distance' in locals() else 0,
        'complexity_score': complexity_score
    }


if __name__ == "__main__":
    # Test level validation
    print("Testing level validation system...")
    
    import sys
    import os
    sys.path.append('.')
    
    # Add ftlib test directory to path
    ftlib_test_path = os.path.join(os.path.dirname(__file__), '..', 'ftlib', 'test')
    sys.path.append(ftlib_test_path)
    
    try:
        from waypoint_test_runner import load_maze_like_levels, create_test_screenshot
        from get_design import retrieveLevel, designDomToStruct
        
        # Load a few levels for testing
        tsv_path = os.path.join(os.path.dirname(__file__), '..', 'maze_like_levels.tsv')
        level_ids = load_maze_like_levels(tsv_path)[:5]  # Test first 5
        
        level_data = []
        for level_id in level_ids:
            try:
                level_dom = retrieveLevel(level_id, is_design=False)
                design_struct = designDomToStruct(level_dom)
                screenshot = create_test_screenshot(design_struct, (200, 145))
                level_data.append((f"level_{level_id}", screenshot))
            except Exception as e:
                print(f"Failed to load level {level_id}: {e}")
        
        print(f"\nLoaded {len(level_data)} levels for validation testing")
        
        # Validate levels
        valid_levels, invalid_levels = validate_level_database(level_data, verbose=True)
        
        print(f"\n=== Validation Results ===")
        print(f"Valid levels: {len(valid_levels)}")
        print(f"Invalid levels: {len(invalid_levels)}")
        
        if invalid_levels:
            print("\nInvalid levels:")
            for name, reason in invalid_levels:
                print(f"  {name}: {reason}")
        
        # Analyze characteristics of first valid level
        if valid_levels:
            name, screenshot = valid_levels[0]
            characteristics = analyze_level_characteristics(screenshot)
            print(f"\n=== Level Characteristics: {name} ===")
            for key, value in characteristics.items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.3f}")
                else:
                    print(f"  {key}: {value}")
    
    except ImportError:
        print("ftlib not available - testing with synthetic levels")
        from waypoint_test_runner import create_synthetic_test_cases
        
        test_cases = create_synthetic_test_cases()
        level_data = [(name, screenshot) for name, screenshot in test_cases]
        
        valid_levels, invalid_levels = validate_level_database(level_data, verbose=True)
        print(f"\nSynthetic level validation: {len(valid_levels)}/{len(level_data)} valid")
    
    print("\nLevel validation testing complete!")