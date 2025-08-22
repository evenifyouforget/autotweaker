"""
Waypoint scoring and validation system for autotweaker.

This module implements scoring functions to evaluate waypoint lists based on:
1. Non-skippability requirements (all paths must go through waypoints in order)
2. Quality metrics (no local valleys, reasonable circle sizes, etc.)
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Set, Optional
from collections import deque
import networkx as nx


def check_waypoint_non_skippability(screenshot: np.ndarray, waypoints: List[Dict[str, float]]) -> bool:
    """
    Check if waypoints meet the non-skippability requirement.
    
    Every possible path from a source to a sink must pass through every waypoint in order.
    
    Args:
        screenshot: 2D array where 1=wall, 3=source, 4=sink, 0=passable
        waypoints: List of waypoint dicts with keys 'x', 'y', 'radius' (in pixel coordinates)
    
    Returns:
        bool: True if waypoints are non-skippable, False otherwise
    """
    if len(waypoints) == 0:
        # Empty waypoint list is always valid (check direct connectivity)
        return _check_basic_connectivity(screenshot)
    
    # For each waypoint, check if removing it (turning it into walls) 
    # breaks connectivity from sources to the next target
    for i in range(len(waypoints)):
        # Create modified screenshot with current waypoint turned into walls
        modified_screenshot = screenshot.copy()
        waypoint = waypoints[i]
        
        # Convert waypoint circle to wall pixels
        _mark_circle_as_wall(modified_screenshot, waypoint['x'], waypoint['y'], waypoint['radius'])
        
        # Determine source and target for this waypoint
        if i == 0:
            # First waypoint: check sources to next waypoint (or sink if last)
            sources = _find_pixels_by_type(screenshot, 3)  # source pixels
            if i == len(waypoints) - 1:
                targets = _find_pixels_by_type(screenshot, 4)  # sink pixels
            else:
                targets = _get_waypoint_pixels(waypoints[i + 1])
        else:
            # Not first waypoint: check previous waypoint to next target
            sources = _get_waypoint_pixels(waypoints[i - 1])
            if i == len(waypoints) - 1:
                targets = _find_pixels_by_type(screenshot, 4)  # sink pixels
            else:
                targets = _get_waypoint_pixels(waypoints[i + 1])
        
        # Check if there's still a path from sources to targets without this waypoint
        if _check_connectivity_between_sets(modified_screenshot, sources, targets):
            # Path exists without this waypoint, so it's skippable
            return False
    
    return True


def score_waypoint_list(screenshot: np.ndarray, waypoints: List[Dict[str, float]], 
                       penalize_skippable: bool = True, feature_flags: Optional[Dict[str, bool]] = None) -> float:
    """
    Score a waypoint list (lower is better).
    
    Args:
        screenshot: 2D array where 1=wall, 3=source, 4=sink, 0=passable
        waypoints: List of waypoint dicts with keys 'x', 'y', 'radius' (in pixel coordinates)
        penalize_skippable: If True, heavily penalize lists that don't meet non-skippability
        feature_flags: Dict of toggleable features for scoring
    
    Returns:
        float: Score (lower is better), negative scores indicate solving designs
    """
    if feature_flags is None:
        feature_flags = {}
    
    score = 0.0
    
    # Check non-skippability requirement
    is_non_skippable = check_waypoint_non_skippability(screenshot, waypoints)
    if penalize_skippable and not is_non_skippable:
        score += 10000.0  # Heavy penalty for skippable waypoints
    
    # Basic connectivity check
    if not _check_basic_connectivity(screenshot):
        score += 5000.0  # Level is unsolvable
    
    # Quality metrics
    if feature_flags.get('check_local_valleys', True):
        score += _score_local_valleys(screenshot, waypoints) * 100.0
    
    if feature_flags.get('check_circle_sizes', True):
        score += _score_circle_sizes(screenshot, waypoints) * 10.0
    
    if feature_flags.get('check_overlaps', True):
        score += _score_overlaps(screenshot, waypoints) * 50.0
    
    if feature_flags.get('check_path_efficiency', True):
        score += _score_path_efficiency(screenshot, waypoints) * 5.0
    
    if feature_flags.get('penalize_excessive_waypoints', True):
        score += len(waypoints) * 1.0  # Small penalty for more waypoints
    
    return score


def _check_basic_connectivity(screenshot: np.ndarray) -> bool:
    """Check if sources can reach sinks in the base screenshot."""
    sources = _find_pixels_by_type(screenshot, 3)
    sinks = _find_pixels_by_type(screenshot, 4)
    
    if len(sources) == 0 or len(sinks) == 0:
        return False
    
    return _check_connectivity_between_sets(screenshot, sources, sinks)


def _find_pixels_by_type(screenshot: np.ndarray, pixel_type: int) -> List[Tuple[int, int]]:
    """Find all pixels of a given type in the screenshot."""
    positions = np.where(screenshot == pixel_type)
    return list(zip(positions[1], positions[0]))  # (x, y) format


def _get_waypoint_pixels(waypoint: Dict[str, float]) -> List[Tuple[int, int]]:
    """Get pixels covered by a waypoint circle."""
    pixels = []
    x, y, radius = waypoint['x'], waypoint['y'], waypoint['radius']
    
    # Get bounding box
    min_px = max(0, int(x - radius - 1))
    max_px = int(x + radius + 2)
    min_py = max(0, int(y - radius - 1))  
    max_py = int(y + radius + 2)
    
    for px in range(min_px, max_px):
        for py in range(min_py, max_py):
            if (px - x) ** 2 + (py - y) ** 2 <= radius ** 2:
                pixels.append((px, py))
    
    return pixels


def _mark_circle_as_wall(screenshot: np.ndarray, x: float, y: float, radius: float):
    """Mark pixels within a circle as walls (type 1)."""
    height, width = screenshot.shape
    
    min_px = max(0, int(x - radius - 1))
    max_px = min(width, int(x + radius + 2))
    min_py = max(0, int(y - radius - 1))
    max_py = min(height, int(y + radius + 2))
    
    for px in range(min_px, max_px):
        for py in range(min_py, max_py):
            if (px - x) ** 2 + (py - y) ** 2 <= radius ** 2:
                screenshot[py, px] = 1  # Mark as wall


def _check_connectivity_between_sets(screenshot: np.ndarray, sources: List[Tuple[int, int]], 
                                   targets: List[Tuple[int, int]]) -> bool:
    """Check if any source can reach any target using BFS."""
    if not sources or not targets:
        return False
    
    height, width = screenshot.shape
    visited = set()
    queue = deque(sources)
    visited.update(sources)
    
    # BFS from all sources
    while queue:
        x, y = queue.popleft()
        
        # Check if we reached a target
        if (x, y) in targets:
            return True
        
        # Explore neighbors (4-directional)
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            # Check bounds
            if nx < 0 or nx >= width or ny < 0 or ny >= height:
                continue
            
            # Check if already visited
            if (nx, ny) in visited:
                continue
            
            # Check if passable (not wall)
            if screenshot[ny, nx] == 1:  # wall
                continue
            
            visited.add((nx, ny))
            queue.append((nx, ny))
    
    return False


def _score_local_valleys(screenshot: np.ndarray, waypoints: List[Dict[str, float]]) -> float:
    """Score for local valley detection (higher score = more valleys = worse)."""
    if len(waypoints) == 0:
        return 0.0
    
    # Sample points in passable areas and check if they're in local valleys
    valleys_found = 0
    total_samples = 0
    
    height, width = screenshot.shape
    
    # Sample points in a grid
    sample_spacing = max(10, min(width, height) // 20)
    for x in range(0, width, sample_spacing):
        for y in range(0, height, sample_spacing):
            if screenshot[y, x] != 1:  # Not a wall
                total_samples += 1
                if _is_local_valley(screenshot, waypoints, x, y):
                    valleys_found += 1
    
    if total_samples == 0:
        return 0.0
    
    return valleys_found / total_samples


def _is_local_valley(screenshot: np.ndarray, waypoints: List[Dict[str, float]], 
                    x: int, y: int) -> bool:
    """Check if a point is in a local valley (locally minimizes distance to next waypoint)."""
    if len(waypoints) == 0:
        return False
    
    # Find which waypoint this point should be heading to
    current_distance = _distance_to_next_waypoint(waypoints, x, y)
    
    # Check neighboring points
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        nx, ny = x + dx, y + dy
        
        # Check bounds and passability
        if (nx >= 0 and nx < screenshot.shape[1] and ny >= 0 and ny < screenshot.shape[0] and
            screenshot[ny, nx] != 1):
            
            neighbor_distance = _distance_to_next_waypoint(waypoints, nx, ny)
            if neighbor_distance < current_distance:
                return False  # Not a local minimum
    
    return True  # Local minimum found (valley)


def _distance_to_next_waypoint(waypoints: List[Dict[str, float]], x: int, y: int) -> float:
    """Calculate distance from point to next waypoint in sequence."""
    if len(waypoints) == 0:
        return 0.0
    
    # For simplicity, assume we're always heading to the first waypoint
    # In reality, this would depend on which waypoints have been reached
    wp = waypoints[0]
    return math.sqrt((x - wp['x'])**2 + (y - wp['y'])**2)


def _score_circle_sizes(screenshot: np.ndarray, waypoints: List[Dict[str, float]]) -> float:
    """Score for circle sizes (penalize extremely large circles)."""
    if len(waypoints) == 0:
        return 0.0
    
    height, width = screenshot.shape
    max_dimension = max(height, width)
    
    penalty = 0.0
    for waypoint in waypoints:
        radius = waypoint['radius']
        # Penalize circles larger than 1/4 of the screen
        if radius > max_dimension / 4:
            penalty += (radius - max_dimension / 4) / max_dimension
    
    return penalty


def _score_overlaps(screenshot: np.ndarray, waypoints: List[Dict[str, float]]) -> float:
    """Score for waypoint overlaps (penalize overlapping circles)."""
    if len(waypoints) <= 1:
        return 0.0
    
    overlap_penalty = 0.0
    
    # Check waypoint-waypoint overlaps
    for i in range(len(waypoints)):
        for j in range(i + 1, len(waypoints)):
            wp1, wp2 = waypoints[i], waypoints[j]
            distance = math.sqrt((wp1['x'] - wp2['x'])**2 + (wp1['y'] - wp2['y'])**2)
            min_distance = wp1['radius'] + wp2['radius']
            
            if distance < min_distance:
                overlap_penalty += (min_distance - distance) / min_distance
    
    # Check waypoint-source/sink overlaps
    sources = _find_pixels_by_type(screenshot, 3)
    sinks = _find_pixels_by_type(screenshot, 4)
    
    for waypoint in waypoints:
        for sx, sy in sources:
            distance = math.sqrt((waypoint['x'] - sx)**2 + (waypoint['y'] - sy)**2)
            if distance < waypoint['radius']:
                overlap_penalty += (waypoint['radius'] - distance) / waypoint['radius']
        
        for sx, sy in sinks:
            distance = math.sqrt((waypoint['x'] - sx)**2 + (waypoint['y'] - sy)**2)
            if distance < waypoint['radius']:
                overlap_penalty += (waypoint['radius'] - distance) / waypoint['radius']
    
    return overlap_penalty


def _score_path_efficiency(screenshot: np.ndarray, waypoints: List[Dict[str, float]]) -> float:
    """Score for path efficiency (penalize unnecessarily long paths)."""
    if len(waypoints) == 0:
        return 0.0
    
    sources = _find_pixels_by_type(screenshot, 3)
    sinks = _find_pixels_by_type(screenshot, 4)
    
    if not sources or not sinks:
        return 0.0
    
    # Calculate waypoint path length
    total_waypoint_distance = 0.0
    
    # From sources to first waypoint
    for sx, sy in sources:
        distance = math.sqrt((waypoints[0]['x'] - sx)**2 + (waypoints[0]['y'] - sy)**2)
        total_waypoint_distance += distance
    
    # Between waypoints
    for i in range(len(waypoints) - 1):
        wp1, wp2 = waypoints[i], waypoints[i + 1]
        distance = math.sqrt((wp1['x'] - wp2['x'])**2 + (wp1['y'] - wp2['y'])**2)
        total_waypoint_distance += distance
    
    # From last waypoint to sinks
    for sx, sy in sinks:
        distance = math.sqrt((waypoints[-1]['x'] - sx)**2 + (waypoints[-1]['y'] - sy)**2)
        total_waypoint_distance += distance
    
    # Calculate direct distance (as baseline)
    direct_distance = 0.0
    for sx, sy in sources:
        for tx, ty in sinks:
            direct_distance += math.sqrt((tx - sx)**2 + (ty - sy)**2)
    
    if direct_distance == 0:
        return 0.0
    
    # Penalize paths that are much longer than direct
    efficiency_ratio = total_waypoint_distance / direct_distance
    if efficiency_ratio > 2.0:  # More than 2x longer
        return efficiency_ratio - 2.0
    
    return 0.0