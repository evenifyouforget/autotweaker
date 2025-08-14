import numpy as np
import logging
import time
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Color constants for level analysis
COLOR_BACKGROUND = 0
COLOR_STATIC = 1
COLOR_DYNAMIC = 2
COLOR_GOAL_PIECE = 3
COLOR_GOAL_AREA = 4

# World coordinate bounds (matches screenshot.py)
WORLD_MIN_X = -2000
WORLD_MAX_X = 2000
WORLD_MIN_Y = -1450
WORLD_MAX_Y = 1450

@dataclass
class GraphNode:
    """Node in the level graph"""
    x: int
    y: int
    world_x: float
    world_y: float
    is_passable: bool
    
class WaypointGenerationError(Exception):
    """Custom error for waypoint generation failures"""
    pass

def generate_waypoints(screenshot: np.ndarray,
                       limit_time_seconds: float | None = None,
                       limit_iterations: int | None = None) -> List[Dict[str, float]]:
    """
    Generate waypoints for a level screenshot using graph-based pathfinding.
    
    Args:
        screenshot: 2D numpy array with integer color values
        limit_time_seconds: Time budget (None for no limit)
        limit_iterations: Iteration budget (None for no limit)
        
    Returns:
        List of waypoint dictionaries with keys 'x', 'y', 'radius' ordered from start to goal
        
    Raises:
        WaypointGenerationError: If generation fails
    """
    start_time = time.time()
    iteration = 0
    
    try:
        logger.debug(f"Starting waypoint generation for {screenshot.shape} image")
        
        # Extract basic level information
        goal_pieces, goal_area, passable_mask = _extract_level_info(screenshot)
        
        if len(goal_pieces) == 0:
            raise WaypointGenerationError("No goal pieces found")
        if goal_area is None:
            raise WaypointGenerationError("No goal area found")
            
        # Convert to world coordinates
        height, width = screenshot.shape
        
        def pixel_to_world(px, py):
            world_x = WORLD_MIN_X + (px + 0.5) * (WORLD_MAX_X - WORLD_MIN_X) / width
            world_y = WORLD_MIN_Y + (py + 0.5) * (WORLD_MAX_Y - WORLD_MIN_Y) / height
            return world_x, world_y
            
        # Main generation algorithm
        best_waypoints = []
        best_score = float('inf')
        
        max_iterations = 100 if limit_iterations is None else limit_iterations
        max_iterations = max(1, max_iterations)  # Ensure at least 1 iteration
        
        for iteration in range(max_iterations):
            # Check time budget
            if limit_time_seconds is not None and limit_time_seconds > 0 and time.time() - start_time > limit_time_seconds:
                logger.debug(f"Time budget exceeded after {iteration} iterations")
                break
                
            # Generate waypoint candidate
            waypoints = _generate_waypoint_candidate(
                goal_pieces, goal_area, passable_mask, pixel_to_world
            )
            
            # Score the waypoints
            score = _score_waypoints(waypoints, goal_pieces, goal_area, pixel_to_world)
            
            if score < best_score:
                best_score = score
                best_waypoints = waypoints
                logger.debug(f"New best waypoints (score {score:.2f}) after {iteration} iterations")
                
            # Early termination if we found a good solution
            if score < 0.1:  # Arbitrary threshold for "good enough"
                break
                
        logger.info(f"Generated {len(best_waypoints)} waypoints in {iteration + 1} iterations")
        return best_waypoints
        
    except Exception as e:
        logger.error(f"Waypoint generation failed: {e}")
        raise WaypointGenerationError(f"Generation failed: {e}") from e

def _extract_level_info(screenshot: np.ndarray) -> Tuple[List[Tuple[int, int]], Optional[Tuple[int, int, int, int]], np.ndarray]:
    """Extract goal pieces, goal area, and passable areas from screenshot"""
    
    # Find goal pieces (color 3)
    goal_piece_coords = np.where(screenshot == COLOR_GOAL_PIECE)
    goal_pieces = [(int(x), int(y)) for y, x in zip(goal_piece_coords[0], goal_piece_coords[1])]
    
    # Find goal area bounds (color 4)
    goal_area_coords = np.where(screenshot == COLOR_GOAL_AREA)
    if len(goal_area_coords[0]) > 0:
        min_y, max_y = goal_area_coords[0].min(), goal_area_coords[0].max()
        min_x, max_x = goal_area_coords[1].min(), goal_area_coords[1].max()
        goal_area = (min_x, min_y, max_x - min_x, max_y - min_y)
    else:
        goal_area = None
        
    # Create passable mask (anything that's not static)
    passable_mask = screenshot != COLOR_STATIC
    
    return goal_pieces, goal_area, passable_mask

def _generate_waypoint_candidate(goal_pieces: List[Tuple[int, int]], 
                                goal_area: Tuple[int, int, int, int], passable_mask: np.ndarray,
                                pixel_to_world) -> List[Dict[str, float]]:
    """Generate a candidate set of waypoints using simple pathfinding"""
    
    if not goal_pieces or goal_area is None:
        return []
        
    # Use the first goal piece as representative (simplification for multiple goal pieces)
    start_x, start_y = goal_pieces[0]
    goal_center_x = goal_area[0] + goal_area[2] // 2
    goal_center_y = goal_area[1] + goal_area[3] // 2
    
    # Simple A* pathfinding implementation
    try:
        path = _find_path(start_x, start_y, goal_center_x, goal_center_y, passable_mask)
        
        if path:
            # Simplify path to key waypoints
            waypoints = _simplify_path_to_waypoints(path, passable_mask, pixel_to_world)
            return waypoints
        else:
            logger.debug("No path found, using fallback")
            
    except Exception as e:
        logger.debug(f"Pathfinding failed: {e}, using fallback")
        
    # Fallback: single waypoint at midpoint
    mid_x = (start_x + goal_center_x) // 2
    mid_y = (start_y + goal_center_y) // 2
    world_x, world_y = pixel_to_world(mid_x, mid_y)
    return [{"x": world_x, "y": world_y, "radius": 100.0}]

def _find_path(start_x: int, start_y: int, goal_x: int, goal_y: int, passable_mask: np.ndarray) -> List[Tuple[int, int]]:
    """Simple A* pathfinding on the passable mask"""
    height, width = passable_mask.shape
    
    if not (0 <= start_x < width and 0 <= start_y < height):
        return []
    if not (0 <= goal_x < width and 0 <= goal_y < height):
        return []
    if not passable_mask[start_y, start_x] or not passable_mask[goal_y, goal_x]:
        return []
    
    # A* implementation
    from heapq import heappush, heappop
    
    def heuristic(x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)
    
    open_set = [(0, start_x, start_y)]
    came_from = {}
    g_score = {(start_x, start_y): 0}
    f_score = {(start_x, start_y): heuristic(start_x, start_y, goal_x, goal_y)}
    visited = set()
    
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    
    while open_set:
        _, current_x, current_y = heappop(open_set)
        
        if (current_x, current_y) in visited:
            continue
        visited.add((current_x, current_y))
        
        if current_x == goal_x and current_y == goal_y:
            # Reconstruct path
            path = []
            x, y = current_x, current_y
            while (x, y) in came_from:
                path.append((x, y))
                x, y = came_from[(x, y)]
            path.append((start_x, start_y))
            return list(reversed(path))
        
        for dx, dy in directions:
            neighbor_x, neighbor_y = current_x + dx, current_y + dy
            
            if not (0 <= neighbor_x < width and 0 <= neighbor_y < height):
                continue
            if not passable_mask[neighbor_y, neighbor_x]:
                continue
            if (neighbor_x, neighbor_y) in visited:
                continue
            
            tentative_g = g_score[(current_x, current_y)] + (1.414 if dx != 0 and dy != 0 else 1.0)
            
            if (neighbor_x, neighbor_y) not in g_score or tentative_g < g_score[(neighbor_x, neighbor_y)]:
                came_from[(neighbor_x, neighbor_y)] = (current_x, current_y)
                g_score[(neighbor_x, neighbor_y)] = tentative_g
                f_score[(neighbor_x, neighbor_y)] = tentative_g + heuristic(neighbor_x, neighbor_y, goal_x, goal_y)
                heappush(open_set, (f_score[(neighbor_x, neighbor_y)], neighbor_x, neighbor_y))
    
    return []  # No path found

def _simplify_path_to_waypoints(path: List[Tuple[int, int]], passable_mask: np.ndarray, 
                               pixel_to_world) -> List[Dict[str, float]]:
    """Convert a pixel path to a simplified set of waypoints"""
    if len(path) < 3:
        return []
        
    waypoints = []
    
    # Use Douglas-Peucker-like algorithm to find key turning points
    def find_turning_points(path):
        if len(path) <= 2:
            return list(range(len(path)))
            
        points = [0]  # Always include start
        
        for i in range(1, len(path) - 1):
            # Check if this point represents a significant direction change
            prev_point = path[i-1]
            curr_point = path[i]
            next_point = path[i+1]
            
            # Calculate direction vectors
            v1 = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
            v2 = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])
            
            # Calculate angle between vectors
            dot_product = v1[0] * v2[0] + v1[1] * v2[1]
            magnitude1 = (v1[0]**2 + v1[1]**2)**0.5
            magnitude2 = (v2[0]**2 + v2[1]**2)**0.5
            
            if magnitude1 > 0 and magnitude2 > 0:
                cos_angle = dot_product / (magnitude1 * magnitude2)
                cos_angle = max(-1, min(1, cos_angle))  # Clamp to valid range
                angle = np.arccos(cos_angle)
                
                # If angle is significant (> 30 degrees), it's a turning point
                if angle > np.pi / 6:
                    points.append(i)
                    
        points.append(len(path) - 1)  # Always include end
        return points
    
    turning_points = find_turning_points(path)
    
    for i in turning_points[:-1]:  # Exclude the last point (goal area)
        px, py = path[i]
        world_x, world_y = pixel_to_world(px, py)
        
        # Calculate appropriate radius based on local corridor width
        radius = _calculate_waypoint_radius(px, py, passable_mask)
        
        waypoints.append({"x": world_x, "y": world_y, "radius": radius})
    
    return waypoints

def _calculate_waypoint_radius(px: int, py: int, passable_mask: np.ndarray) -> float:
    """Calculate appropriate radius for a waypoint based on local corridor width"""
    height, width = passable_mask.shape
    
    # Find distance to nearest wall in multiple directions
    directions = [(0, 1), (1, 0), (0, -1), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]
    min_distance = float('inf')
    
    for dx, dy in directions:
        distance = 0
        x, y = px, py
        
        while 0 <= x < width and 0 <= y < height and passable_mask[y, x]:
            x += dx
            y += dy
            distance += 1
            if distance > 50:  # Prevent infinite search
                break
                
        min_distance = min(min_distance, distance)
    
    # Convert pixel distance to world units
    world_width = WORLD_MAX_X - WORLD_MIN_X
    world_height = WORLD_MAX_Y - WORLD_MIN_Y
    pixel_to_world_scale = min(world_width / width, world_height / height)
    
    # Radius should be 70% of corridor width but at least 50 world units
    radius = max(50.0, min_distance * pixel_to_world_scale * 0.7)
    
    return radius

def _score_waypoints(waypoints: List[Dict[str, float]], goal_pieces: List[Tuple[int, int]], 
                    goal_area: Tuple[int, int, int, int], pixel_to_world) -> float:
    """Score a set of waypoints based on various criteria"""
    if not waypoints:
        return float('inf')
        
    score = 0.0
    
    # Penalty for too many waypoints
    score += len(waypoints) * 0.1
    
    # Penalty for waypoints that are too close together
    for i in range(len(waypoints) - 1):
        w1, w2 = waypoints[i], waypoints[i + 1]
        distance = ((w1["x"] - w2["x"])**2 + (w1["y"] - w2["y"])**2)**0.5
        if distance < 200:  # Too close
            score += (200 - distance) * 0.01
            
    # Bonus for waypoints that form a reasonable path
    if len(goal_pieces) > 0:
        gp_x, gp_y = goal_pieces[0]
        gp_world_x, gp_world_y = pixel_to_world(gp_x, gp_y)
        goal_world_x, goal_world_y = pixel_to_world(
            goal_area[0] + goal_area[2] // 2,
            goal_area[1] + goal_area[3] // 2
        )
        
        # Calculate total path length
        path_length = 0.0
        current_x, current_y = gp_world_x, gp_world_y
        
        for waypoint in waypoints:
            path_length += ((waypoint["x"] - current_x)**2 + (waypoint["y"] - current_y)**2)**0.5
            current_x, current_y = waypoint["x"], waypoint["y"]
            
        path_length += ((goal_world_x - current_x)**2 + (goal_world_y - current_y)**2)**0.5
        
        # Compare to direct distance
        direct_distance = ((goal_world_x - gp_world_x)**2 + (goal_world_y - gp_world_y)**2)**0.5
        
        if direct_distance > 0:
            path_efficiency = direct_distance / path_length
            score += (1.0 - path_efficiency) * 10.0  # Penalty for inefficient paths
    
    return score