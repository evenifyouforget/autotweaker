"""
Fast CornerTurning variants that optimize the expensive parts with heuristics and math tricks.

Tests different approaches to see if simplified CornerTurning performs as well as the full version:
1. FastCornerTurning: Skip balloon expansion, use fixed radius
2. HeuristicCornerTurning: Use distance-based heuristics instead of pathfinding
3. GeometricCornerTurning: Use purely geometric corner detection
4. MinimalCornerTurning: Most stripped-down version
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional
from collections import deque

try:
    from .waypoint_generation import WaypointGenerator
except ImportError:
    from waypoint_generation import WaypointGenerator


class FastCornerTurningGenerator(WaypointGenerator):
    """
    Fast CornerTurning that skips expensive balloon expansion.
    
    Uses fixed radius and simpler validation instead of iterative balloon expansion.
    """
    
    def __init__(self, max_waypoints: int = 20, fixed_radius: float = 15.0):
        super().__init__("FastCornerTurning")
        self.max_waypoints = max_waypoints
        self.fixed_radius = fixed_radius
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using fast corner-turning algorithm."""
        sources = self._find_pixels_by_type(screenshot, 3)
        sinks = self._find_pixels_by_type(screenshot, 4)
        
        if not sources or not sinks:
            return []
        
        waypoints = self._fast_corner_algorithm(screenshot, sources, sinks)
        return waypoints[:self.max_waypoints]
    
    def _fast_corner_algorithm(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], 
                              targets: List[Tuple[int, int]]) -> List[Dict[str, float]]:
        """Fast corner algorithm with minimal pathfinding."""
        
        # Quick line-of-sight check
        if self._has_direct_line_of_sight(screenshot, sources, targets):
            return []
        
        # Simple path finding using straight line + correction
        path = self._fast_path_find(screenshot, sources, targets)
        if not path or len(path) < 3:
            return []
        
        # Find corner point (simplified)
        corner_point = self._find_simple_corner(screenshot, path, targets)
        if corner_point is None:
            return []
        
        # Create waypoint with fixed radius (no balloon expansion)
        waypoint = {
            'x': float(corner_point[0]),
            'y': float(corner_point[1]), 
            'radius': self.fixed_radius
        }
        
        # Quick blocking check (simplified)
        if not self._quick_blocking_check(screenshot, waypoint, sources, targets):
            return []
        
        # Recurse (depth limited to prevent infinite recursion)
        if len(path) > 10:  # Only recurse for longer paths
            recursive_waypoints = self._fast_corner_algorithm(
                screenshot, sources, [(int(waypoint['x']), int(waypoint['y']))]
            )
            return recursive_waypoints + [waypoint]
        
        return [waypoint]
    
    def _fast_path_find(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], 
                       targets: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Fast pathfinding using greedy approach."""
        if not sources or not targets:
            return []
        
        # Start from closest source-target pair
        best_dist = float('inf')
        best_source = sources[0]
        best_target = targets[0]
        
        for source in sources:
            for target in targets:
                dist = math.sqrt((source[0] - target[0])**2 + (source[1] - target[1])**2)
                if dist < best_dist:
                    best_dist = dist
                    best_source = source
                    best_target = target
        
        # Simple greedy pathfinding
        path = [best_source]
        current = best_source
        
        while current != best_target:
            # Find next step toward target
            dx = best_target[0] - current[0]
            dy = best_target[1] - current[1]
            
            # Normalize to unit step
            if abs(dx) > abs(dy):
                step = (1 if dx > 0 else -1, 0)
            else:
                step = (0, 1 if dy > 0 else -1)
            
            next_pos = (current[0] + step[0], current[1] + step[1])
            
            # Check bounds and walls
            if (next_pos[0] < 0 or next_pos[0] >= screenshot.shape[1] or
                next_pos[1] < 0 or next_pos[1] >= screenshot.shape[0] or
                screenshot[next_pos[1], next_pos[0]] == 1):
                # Hit obstacle - try different direction
                if abs(dx) > abs(dy):
                    step = (0, 1 if dy > 0 else -1)
                else:
                    step = (1 if dx > 0 else -1, 0)
                
                next_pos = (current[0] + step[0], current[1] + step[1])
                
                # If still blocked, give up
                if (next_pos[0] < 0 or next_pos[0] >= screenshot.shape[1] or
                    next_pos[1] < 0 or next_pos[1] >= screenshot.shape[0] or
                    screenshot[next_pos[1], next_pos[0]] == 1):
                    break
            
            current = next_pos
            path.append(current)
            
            # Prevent infinite loops
            if len(path) > 200:
                break
        
        return path
    
    def _find_simple_corner(self, screenshot: np.ndarray, path: List[Tuple[int, int]], 
                           targets: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find corner point using simple heuristic."""
        if len(path) < 3:
            return None
        
        # Look for direction changes in path
        for i in range(1, len(path) - 1):
            prev_point = path[i - 1]
            curr_point = path[i]
            next_point = path[i + 1]
            
            # Check if direction changed significantly
            vec1 = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
            vec2 = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])
            
            # Dot product to check angle
            if vec1 != (0, 0) and vec2 != (0, 0):
                dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
                if dot <= 0:  # 90+ degree turn
                    return curr_point
        
        # Fallback: return middle of path
        return path[len(path) // 2]
    
    def _quick_blocking_check(self, screenshot: np.ndarray, waypoint: Dict[str, float],
                             sources: List[Tuple[int, int]], targets: List[Tuple[int, int]]) -> bool:
        """Quick check if waypoint blocks paths (simplified version)."""
        # Just check if waypoint is between sources and targets
        wx, wy = waypoint['x'], waypoint['y']
        
        for source in sources:
            for target in targets:
                # Check if waypoint is roughly on the line between source and target
                sx, sy = source
                tx, ty = target
                
                # Distance from waypoint to line
                A = ty - sy
                B = sx - tx
                C = tx * sy - sx * ty
                
                if A != 0 or B != 0:
                    line_dist = abs(A * wx + B * wy + C) / math.sqrt(A**2 + B**2)
                    if line_dist < waypoint['radius']:
                        return True
        
        return False
    
    # Reuse helper methods from original
    def _find_pixels_by_type(self, screenshot: np.ndarray, pixel_type: int) -> List[Tuple[int, int]]:
        """Find all pixels of a given type."""
        positions = np.where(screenshot == pixel_type)
        return list(zip(positions[1], positions[0]))  # (x, y) format
    
    def _has_direct_line_of_sight(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], 
                                targets: List[Tuple[int, int]]) -> bool:
        """Check if any source has direct line-of-sight to any target."""
        for sx, sy in sources:
            for tx, ty in targets:
                if self._line_of_sight_clear(screenshot, sx, sy, tx, ty):
                    return True
        return False
    
    def _line_of_sight_clear(self, screenshot: np.ndarray, x1: int, y1: int, 
                           x2: int, y2: int) -> bool:
        """Check if line-of-sight is clear between two points."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        x_inc = 1 if x1 < x2 else -1
        y_inc = 1 if y1 < y2 else -1
        error = dx - dy
        
        while True:
            if (x < 0 or x >= screenshot.shape[1] or y < 0 or y >= screenshot.shape[0] or
                screenshot[y, x] == 1):
                return False
            
            if x == x2 and y == y2:
                break
            
            error2 = 2 * error
            if error2 > -dy:
                error -= dy
                x += x_inc
            if error2 < dx:
                error += dx
                y += y_inc
        
        return True


class HeuristicCornerTurningGenerator(WaypointGenerator):
    """
    CornerTurning using distance-based heuristics instead of pathfinding.
    
    Uses mathematical tricks to estimate where corners should be placed.
    """
    
    def __init__(self, max_waypoints: int = 20):
        super().__init__("HeuristicCornerTurning")
        self.max_waypoints = max_waypoints
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using heuristic-based approach."""
        sources = self._find_pixels_by_type(screenshot, 3)
        sinks = self._find_pixels_by_type(screenshot, 4)
        
        if not sources or not sinks:
            return []
        
        # Use distance field to find likely waypoint locations
        waypoints = self._heuristic_waypoint_placement(screenshot, sources, sinks)
        return waypoints[:self.max_waypoints]
    
    def _heuristic_waypoint_placement(self, screenshot: np.ndarray, 
                                    sources: List[Tuple[int, int]], 
                                    sinks: List[Tuple[int, int]]) -> List[Dict[str, float]]:
        """Place waypoints using distance field heuristics."""
        height, width = screenshot.shape
        
        # Create distance field from sinks
        distance_field = np.full((height, width), float('inf'))
        
        # Initialize sinks
        for sx, sy in sinks:
            distance_field[sy, sx] = 0
        
        # Simple distance propagation (not full Dijkstra for speed)
        changed = True
        iterations = 0
        while changed and iterations < 50:  # Limited iterations for speed
            changed = False
            for y in range(height):
                for x in range(width):
                    if screenshot[y, x] == 1:  # Wall
                        continue
                    
                    current_dist = distance_field[y, x]
                    
                    # Check neighbors
                    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            neighbor_dist = distance_field[ny, nx] + 1
                            if neighbor_dist < current_dist:
                                distance_field[y, x] = neighbor_dist
                                changed = True
            iterations += 1
        
        # Find local maxima in distance field (far from sinks)
        # These are likely corner locations
        waypoint_candidates = []
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if screenshot[y, x] == 1:  # Skip walls
                    continue
                
                current_dist = distance_field[y, x]
                if current_dist == float('inf'):
                    continue
                
                # Check if this is a local maximum
                is_local_max = True
                for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                    nx, ny = x + dx, y + dy
                    if distance_field[ny, nx] > current_dist:
                        is_local_max = False
                        break
                
                if is_local_max and current_dist > 3:  # Minimum distance threshold
                    waypoint_candidates.append((x, y, current_dist))
        
        # Sort by distance (farthest from sinks first)
        waypoint_candidates.sort(key=lambda w: w[2], reverse=True)
        
        # Convert to waypoints
        waypoints = []
        for x, y, dist in waypoint_candidates[:self.max_waypoints]:
            # Radius based on distance - farther from sinks = larger radius
            radius = min(20.0, max(8.0, dist * 1.5))
            
            waypoint = {
                'x': float(x),
                'y': float(y),
                'radius': radius
            }
            waypoints.append(waypoint)
        
        return waypoints
    
    def _find_pixels_by_type(self, screenshot: np.ndarray, pixel_type: int) -> List[Tuple[int, int]]:
        """Find all pixels of a given type."""
        positions = np.where(screenshot == pixel_type)
        return list(zip(positions[1], positions[0]))  # (x, y) format


class GeometricCornerTurningGenerator(WaypointGenerator):
    """
    Purely geometric corner detection using shape analysis.
    
    Finds corners in the level geometry and places waypoints there.
    """
    
    def __init__(self, max_waypoints: int = 20):
        super().__init__("GeometricCornerTurning")
        self.max_waypoints = max_waypoints
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints at geometric corners."""
        sources = self._find_pixels_by_type(screenshot, 3)
        sinks = self._find_pixels_by_type(screenshot, 4)
        
        if not sources or not sinks:
            return []
        
        # Find geometric corners
        corners = self._find_geometric_corners(screenshot)
        
        # Filter corners by relevance to source-sink paths
        relevant_corners = self._filter_relevant_corners(screenshot, corners, sources, sinks)
        
        # Convert to waypoints
        waypoints = []
        for x, y in relevant_corners[:self.max_waypoints]:
            waypoint = {
                'x': float(x),
                'y': float(y),
                'radius': 12.0  # Fixed moderate radius
            }
            waypoints.append(waypoint)
        
        return waypoints
    
    def _find_geometric_corners(self, screenshot: np.ndarray) -> List[Tuple[int, int]]:
        """Find corners in level geometry using convolution."""
        height, width = screenshot.shape
        
        # Corner detection kernels
        corner_patterns = [
            # L-shapes (4 orientations)
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
        ]
        
        corners = []
        
        for pattern in corner_patterns:
            # Convolve pattern with screenshot
            for y in range(1, height - 1):
                for x in range(1, width - 1):
                    # Extract 3x3 patch
                    patch = screenshot[y-1:y+2, x-1:x+2]
                    
                    # Check if pattern matches
                    if patch.shape == (3, 3):
                        match = True
                        for py in range(3):
                            for px in range(3):
                                expected = pattern[py, px]
                                actual = 1 if patch[py, px] == 1 else 0
                                if expected == 1 and actual != 1:
                                    match = False
                                    break
                            if not match:
                                break
                        
                        if match and screenshot[y, x] != 1:  # Center must be passable
                            corners.append((x, y))
        
        return corners
    
    def _filter_relevant_corners(self, screenshot: np.ndarray, corners: List[Tuple[int, int]],
                                sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Filter corners by relevance to source-sink paths."""
        if not corners:
            return []
        
        # Score corners by distance to sources and sinks
        scored_corners = []
        
        for cx, cy in corners:
            # Distance to nearest source
            min_source_dist = min(math.sqrt((cx - sx)**2 + (cy - sy)**2) for sx, sy in sources)
            
            # Distance to nearest sink
            min_sink_dist = min(math.sqrt((cx - sx)**2 + (cy - sy)**2) for sx, sy in sinks)
            
            # Prefer corners that are intermediate distance from both
            # Not too close, not too far
            source_score = 1.0 / (1.0 + abs(min_source_dist - 20))
            sink_score = 1.0 / (1.0 + abs(min_sink_dist - 20))
            
            total_score = source_score + sink_score
            scored_corners.append((cx, cy, total_score))
        
        # Sort by score
        scored_corners.sort(key=lambda c: c[2], reverse=True)
        
        return [(x, y) for x, y, score in scored_corners]
    
    def _find_pixels_by_type(self, screenshot: np.ndarray, pixel_type: int) -> List[Tuple[int, int]]:
        """Find all pixels of a given type."""
        positions = np.where(screenshot == pixel_type)
        return list(zip(positions[1], positions[0]))  # (x, y) format


class MinimalCornerTurningGenerator(WaypointGenerator):
    """
    Most stripped-down CornerTurning variant.
    
    Just places a single waypoint at the geometric center between sources and sinks,
    with some obstacle avoidance.
    """
    
    def __init__(self):
        super().__init__("MinimalCornerTurning")
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate minimal waypoints using simplest possible logic."""
        sources = self._find_pixels_by_type(screenshot, 3)
        sinks = self._find_pixels_by_type(screenshot, 4)
        
        if not sources or not sinks:
            return []
        
        # Find center point between sources and sinks
        source_center_x = sum(sx for sx, sy in sources) / len(sources)
        source_center_y = sum(sy for sx, sy in sources) / len(sources)
        
        sink_center_x = sum(sx for sx, sy in sinks) / len(sinks)
        sink_center_y = sum(sy for sx, sy in sinks) / len(sinks)
        
        # Midpoint
        mid_x = (source_center_x + sink_center_x) / 2
        mid_y = (source_center_y + sink_center_y) / 2
        
        # Find nearest passable location to midpoint
        waypoint_pos = self._find_nearest_passable(screenshot, int(mid_x), int(mid_y))
        
        if waypoint_pos is None:
            return []
        
        waypoint = {
            'x': float(waypoint_pos[0]),
            'y': float(waypoint_pos[1]),
            'radius': 15.0
        }
        
        return [waypoint]
    
    def _find_nearest_passable(self, screenshot: np.ndarray, target_x: int, target_y: int) -> Optional[Tuple[int, int]]:
        """Find nearest passable location to target."""
        height, width = screenshot.shape
        
        # Check target itself first
        if (0 <= target_x < width and 0 <= target_y < height and 
            screenshot[target_y, target_x] != 1):
            return (target_x, target_y)
        
        # Spiral search outward
        for radius in range(1, 20):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    if abs(dx) == radius or abs(dy) == radius:  # Only check perimeter
                        x, y = target_x + dx, target_y + dy
                        if (0 <= x < width and 0 <= y < height and 
                            screenshot[y, x] != 1):
                            return (x, y)
        
        return None
    
    def _find_pixels_by_type(self, screenshot: np.ndarray, pixel_type: int) -> List[Tuple[int, int]]:
        """Find all pixels of a given type."""
        positions = np.where(screenshot == pixel_type)
        return list(zip(positions[1], positions[0]))  # (x, y) format


def create_fast_corner_tournament():
    """Create tournament with all fast CornerTurning variants."""
    try:
        from .waypoint_generation import WaypointTournament
    except ImportError:
        from waypoint_generation import WaypointTournament
    
    tournament = WaypointTournament()
    
    # Add original for comparison
    try:
        from .waypoint_generation import CornerTurningGenerator
        tournament.add_generator(CornerTurningGenerator())
    except ImportError:
        from waypoint_generation import CornerTurningGenerator
        tournament.add_generator(CornerTurningGenerator())
    
    # Add fast variants
    tournament.add_generator(FastCornerTurningGenerator())
    tournament.add_generator(HeuristicCornerTurningGenerator())
    tournament.add_generator(GeometricCornerTurningGenerator())
    tournament.add_generator(MinimalCornerTurningGenerator())
    
    return tournament


if __name__ == "__main__":
    # Test fast CornerTurning variants
    print("Testing fast CornerTurning variants...")
    
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    from waypoint_test_runner import create_synthetic_test_cases
    
    test_cases = create_synthetic_test_cases()
    
    # Create tournament
    tournament = create_fast_corner_tournament()
    
    print(f"Testing {len(tournament.generators)} CornerTurning variants:")
    for gen in tournament.generators:
        print(f"  - {gen.name}")
    
    # Run mini tournament
    results = tournament.run_tournament(test_cases)
    tournament.print_results(results)
    
    print("\nFast CornerTurning variant testing completed!")