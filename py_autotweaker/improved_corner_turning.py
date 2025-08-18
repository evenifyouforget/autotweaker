"""
Improved corner turning algorithm with proper balloon expansion and spring attraction.

This implements the corner turning algorithm as described in the research instructions
with full balloon expansion mechanics and spring-based attraction to corners.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
import random

try:
    from .waypoint_generation import WaypointGenerator
    from .improved_waypoint_scoring import improved_score_waypoint_list
except ImportError:
    from waypoint_generation import WaypointGenerator
    from improved_waypoint_scoring import improved_score_waypoint_list


class ImprovedCornerTurningGenerator(WaypointGenerator):
    """
    Improved corner turning algorithm with proper balloon expansion.
    
    Implements the algorithm exactly as described in the research instructions:
    1. Find path from sinks to sources
    2. Detect first obscured point (corner)  
    3. Balloon expansion with wall repulsion and corner attraction
    4. Validate waypoint blocks all paths
    5. Recurse backwards
    """
    
    def __init__(self, max_iterations: int = 100, max_waypoints: int = 15, 
                 balloon_iterations: int = 100, spring_strength: float = 0.1,
                 wall_repulsion: float = 2.0):
        super().__init__("ImprovedCornerTurning")
        self.max_iterations = max_iterations
        self.max_waypoints = max_waypoints
        self.balloon_iterations = balloon_iterations
        self.spring_strength = spring_strength
        self.wall_repulsion = wall_repulsion
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using improved corner-turning algorithm."""
        sources = self._find_pixels_by_type(screenshot, 3)
        sinks = self._find_pixels_by_type(screenshot, 4)
        
        if not sources or not sinks:
            return []
        
        # Start recursive algorithm
        waypoints = self._corner_algorithm_recursive(screenshot, sources, sinks, 0)
        return waypoints[:self.max_waypoints]  # Limit waypoint count
    
    def _corner_algorithm_recursive(self, screenshot: np.ndarray, 
                                  sources: List[Tuple[int, int]], 
                                  targets: List[Tuple[int, int]],
                                  recursion_depth: int) -> List[Dict[str, float]]:
        """
        Recursive corner-turning algorithm implementation.
        
        Returns waypoints in order from first to last.
        """
        if recursion_depth > self.max_waypoints:
            return []
        
        # Step 1: Check if direct line-of-sight exists from all sources to any target
        if self._all_sources_can_reach_targets_directly(screenshot, sources, targets):
            return []  # No waypoint needed
        
        # Step 2: Find a path from targets back to sources
        path = self._find_path_targets_to_sources(screenshot, targets, sources)
        if not path:
            return []  # No path exists - level may be unsolvable
        
        # Step 3: Find first obscured point along the path
        corner_point = self._find_first_obscured_point(screenshot, path, targets)
        if corner_point is None:
            return []  # No corner found
        
        # Step 4: Create waypoint using balloon expansion
        waypoint = self._create_balloon_waypoint(screenshot, corner_point, targets)
        if waypoint is None:
            return []  # Could not create valid waypoint
        
        # Step 5: Validate waypoint blocks all paths
        if not self._waypoint_blocks_all_paths(screenshot, waypoint, sources, targets):
            return []  # Waypoint is skippable
        
        # Step 6: Recurse - find waypoints from sources to this waypoint
        waypoint_pixels = self._get_waypoint_edge_pixels(waypoint)
        recursive_waypoints = self._corner_algorithm_recursive(
            screenshot, sources, waypoint_pixels, recursion_depth + 1
        )
        
        # Return waypoints in order: recursive first, then this waypoint
        return recursive_waypoints + [waypoint]
    
    def _all_sources_can_reach_targets_directly(self, screenshot: np.ndarray,
                                              sources: List[Tuple[int, int]], 
                                              targets: List[Tuple[int, int]]) -> bool:
        """Check if all sources have direct line-of-sight to at least one target."""
        for source in sources:
            can_reach_any_target = False
            for target in targets:
                if self._line_of_sight_clear(screenshot, source[0], source[1], target[0], target[1]):
                    can_reach_any_target = True
                    break
            if not can_reach_any_target:
                return False
        return True
    
    def _find_path_targets_to_sources(self, screenshot: np.ndarray, 
                                    targets: List[Tuple[int, int]],
                                    sources: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Find a path from targets to sources using BFS (working backwards)."""
        if not targets or not sources:
            return []
        
        height, width = screenshot.shape
        visited = {}
        queue = deque()
        
        # Start BFS from all targets
        for target in targets:
            queue.append(target)
            visited[target] = None
        
        # BFS to find any source
        while queue:
            x, y = queue.popleft()
            
            # Check if we reached any source
            if (x, y) in sources:
                # Reconstruct path from source back to target
                path = []
                current = (x, y)
                while current is not None:
                    path.append(current)
                    current = visited[current]
                return path  # Path from source to target
            
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
                if screenshot[ny, nx] == 1:
                    continue
                
                visited[(nx, ny)] = (x, y)
                queue.append((nx, ny))
        
        return []
    
    def _find_first_obscured_point(self, screenshot: np.ndarray, 
                                 path: List[Tuple[int, int]],
                                 targets: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """
        Find first point on path that cannot see any target directly.
        
        Working along path from source towards targets.
        """
        for point in path:  # Path is source->...->target, so iterate forward
            has_line_of_sight = False
            for target in targets:
                if self._line_of_sight_clear(screenshot, point[0], point[1], target[0], target[1]):
                    has_line_of_sight = True
                    break
            
            if not has_line_of_sight:
                return point  # First obscured point (corner detected)
        
        return None
    
    def _create_balloon_waypoint(self, screenshot: np.ndarray, 
                               corner_point: Tuple[int, int],
                               targets: List[Tuple[int, int]]) -> Optional[Dict[str, float]]:
        """
        Create waypoint using balloon expansion with wall repulsion and corner attraction.
        
        The waypoint starts at the corner and expands like a balloon, repelled by walls
        and attracted back to the original corner position.
        """
        corner_x, corner_y = corner_point
        
        # Start with small waypoint
        best_waypoint = {
            'x': float(corner_x),
            'y': float(corner_y), 
            'radius': 3.0
        }
        
        # Check if starting position is valid
        if not self._is_waypoint_position_valid(screenshot, best_waypoint):
            # Try nearby positions
            for offset in range(1, 5):
                for dx in [-offset, 0, offset]:
                    for dy in [-offset, 0, offset]:
                        if dx == 0 and dy == 0:
                            continue
                        test_waypoint = {
                            'x': float(corner_x + dx),
                            'y': float(corner_y + dy),
                            'radius': 3.0
                        }
                        if self._is_waypoint_position_valid(screenshot, test_waypoint):
                            best_waypoint = test_waypoint
                            break
                    if best_waypoint != {'x': float(corner_x), 'y': float(corner_y), 'radius': 3.0}:
                        break
                if best_waypoint != {'x': float(corner_x), 'y': float(corner_y), 'radius': 3.0}:
                    break
        
        current_waypoint = best_waypoint.copy()
        
        # Balloon expansion iterations
        for iteration in range(self.balloon_iterations):
            # Calculate forces
            wall_force_x, wall_force_y = self._calculate_wall_repulsion_force(screenshot, current_waypoint)
            spring_force_x = self.spring_strength * (corner_x - current_waypoint['x'])
            spring_force_y = self.spring_strength * (corner_y - current_waypoint['y'])
            
            # Apply forces to position
            new_x = current_waypoint['x'] + wall_force_x * 0.1 + spring_force_x
            new_y = current_waypoint['y'] + wall_force_y * 0.1 + spring_force_y
            
            # Try to expand radius slightly
            expansion_force = self._calculate_expansion_pressure(screenshot, current_waypoint)
            new_radius = current_waypoint['radius'] + expansion_force * 0.1
            new_radius = max(2.0, min(new_radius, 25.0))  # Clamp radius
            
            # Create candidate waypoint
            candidate_waypoint = {
                'x': float(new_x),
                'y': float(new_y),
                'radius': new_radius
            }
            
            # Check if candidate is valid and better
            if (self._is_waypoint_position_valid(screenshot, candidate_waypoint) and
                self._waypoint_does_not_obscure_targets(screenshot, candidate_waypoint, targets)):
                
                # Accept if it's an improvement or with some probability for exploration
                if (self._waypoint_quality(screenshot, candidate_waypoint) > 
                    self._waypoint_quality(screenshot, current_waypoint) or
                    random.random() < 0.1):  # 10% exploration
                    current_waypoint = candidate_waypoint
                    
                    # Update best if this is better
                    if (self._waypoint_quality(screenshot, current_waypoint) > 
                        self._waypoint_quality(screenshot, best_waypoint)):
                        best_waypoint = current_waypoint.copy()
        
        # Validate final waypoint
        if (best_waypoint['radius'] >= 2.0 and 
            self._is_waypoint_position_valid(screenshot, best_waypoint)):
            return best_waypoint
        
        return None
    
    def _calculate_wall_repulsion_force(self, screenshot: np.ndarray, 
                                      waypoint: Dict[str, float]) -> Tuple[float, float]:
        """Calculate repulsive force from nearby walls."""
        x, y, radius = waypoint['x'], waypoint['y'], waypoint['radius']
        force_x, force_y = 0.0, 0.0
        
        # Check points around the waypoint circle
        num_samples = 16
        for i in range(num_samples):
            angle = 2 * math.pi * i / num_samples
            sample_x = x + radius * math.cos(angle)
            sample_y = y + radius * math.sin(angle)
            
            # Find distance to nearest wall
            wall_distance = self._distance_to_nearest_wall(screenshot, sample_x, sample_y)
            
            if wall_distance < 5.0:  # Close to wall
                # Calculate repulsion force away from wall direction
                wall_normal_x, wall_normal_y = self._get_wall_normal(screenshot, sample_x, sample_y)
                
                # Force magnitude inversely proportional to distance
                force_magnitude = self.wall_repulsion / (wall_distance + 0.1)
                
                force_x += force_magnitude * wall_normal_x
                force_y += force_magnitude * wall_normal_y
        
        return force_x, force_y
    
    def _calculate_expansion_pressure(self, screenshot: np.ndarray, 
                                    waypoint: Dict[str, float]) -> float:
        """Calculate pressure for waypoint to expand or contract."""
        x, y, radius = waypoint['x'], waypoint['y'], waypoint['radius']
        
        # Check if expansion would hit walls
        expansion_test_radius = radius + 2.0
        
        num_samples = 12
        wall_contacts = 0
        
        for i in range(num_samples):
            angle = 2 * math.pi * i / num_samples
            test_x = x + expansion_test_radius * math.cos(angle)
            test_y = y + expansion_test_radius * math.sin(angle)
            
            if self._is_wall_at_position(screenshot, test_x, test_y):
                wall_contacts += 1
        
        # If many wall contacts, contract; if few, expand
        contact_ratio = wall_contacts / num_samples
        if contact_ratio > 0.3:
            return -1.0  # Contract
        elif contact_ratio < 0.1:
            return 1.0   # Expand
        else:
            return 0.0   # Stay same size
    
    def _distance_to_nearest_wall(self, screenshot: np.ndarray, x: float, y: float) -> float:
        """Find distance to nearest wall from a point."""
        height, width = screenshot.shape
        
        # Check in expanding squares around the point
        for radius in range(1, 20):
            found_wall = False
            
            # Check perimeter of square
            for dx in range(-radius, radius + 1):
                for dy in [-radius, radius]:  # Top and bottom edges
                    test_x, test_y = int(x + dx), int(y + dy)
                    if (0 <= test_x < width and 0 <= test_y < height and 
                        screenshot[test_y, test_x] == 1):
                        return radius
                        
            for dy in range(-radius + 1, radius):
                for dx in [-radius, radius]:  # Left and right edges
                    test_x, test_y = int(x + dx), int(y + dy)
                    if (0 <= test_x < width and 0 <= test_y < height and 
                        screenshot[test_y, test_x] == 1):
                        return radius
        
        return 20.0  # Max distance if no wall found nearby
    
    def _get_wall_normal(self, screenshot: np.ndarray, x: float, y: float) -> Tuple[float, float]:
        """Get normal vector pointing away from nearest wall."""
        # Simple gradient-based normal calculation
        height, width = screenshot.shape
        ix, iy = int(x), int(y)
        
        # Calculate gradient
        grad_x = grad_y = 0.0
        
        if 0 < ix < width - 1 and 0 < iy < height - 1:
            grad_x = float(screenshot[iy, ix + 1] - screenshot[iy, ix - 1])
            grad_y = float(screenshot[iy + 1, ix] - screenshot[iy - 1, ix])
            
            # Normalize
            grad_mag = math.sqrt(grad_x**2 + grad_y**2) + 0.001
            grad_x /= grad_mag
            grad_y /= grad_mag
        
        return -grad_x, -grad_y  # Normal points away from walls
    
    def _is_wall_at_position(self, screenshot: np.ndarray, x: float, y: float) -> bool:
        """Check if there's a wall at the given position."""
        height, width = screenshot.shape
        ix, iy = int(x + 0.5), int(y + 0.5)
        
        if 0 <= ix < width and 0 <= iy < height:
            return screenshot[iy, ix] == 1
        return True  # Out of bounds counts as wall
    
    def _waypoint_quality(self, screenshot: np.ndarray, waypoint: Dict[str, float]) -> float:
        """Estimate waypoint quality (higher is better)."""
        x, y, radius = waypoint['x'], waypoint['y'], waypoint['radius']
        
        # Quality based on distance from walls and reasonable size
        min_wall_distance = self._distance_to_nearest_wall(screenshot, x, y)
        
        # Good waypoints are reasonably sized and not too close to walls
        size_quality = 1.0 / (abs(radius - 8.0) + 1.0)  # Prefer radius around 8
        wall_distance_quality = min(min_wall_distance / 5.0, 1.0)  # Prefer distance >= 5
        
        return size_quality * wall_distance_quality
    
    def _is_waypoint_position_valid(self, screenshot: np.ndarray, waypoint: Dict[str, float]) -> bool:
        """Check if waypoint position is valid (doesn't intersect walls)."""
        x, y, radius = waypoint['x'], waypoint['y'], waypoint['radius']
        height, width = screenshot.shape
        
        # Check circle doesn't intersect with walls
        num_samples = 12
        for i in range(num_samples):
            angle = 2 * math.pi * i / num_samples
            test_x = x + radius * math.cos(angle)
            test_y = y + radius * math.sin(angle)
            
            if self._is_wall_at_position(screenshot, test_x, test_y):
                return False
        
        # Check center area
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if self._is_wall_at_position(screenshot, x + dx, y + dy):
                    return False
        
        return True
    
    def _waypoint_does_not_obscure_targets(self, screenshot: np.ndarray,
                                         waypoint: Dict[str, float],
                                         targets: List[Tuple[int, int]]) -> bool:
        """Check that waypoint doesn't block line of sight to targets."""
        # For simplicity, just check that targets are not inside the waypoint
        x, y, radius = waypoint['x'], waypoint['y'], waypoint['radius']
        
        for tx, ty in targets:
            distance = math.sqrt((tx - x)**2 + (ty - y)**2)
            if distance <= radius:
                return False  # Target is inside waypoint
        
        return True
    
    def _waypoint_blocks_all_paths(self, screenshot: np.ndarray, waypoint: Dict[str, float],
                                 sources: List[Tuple[int, int]], 
                                 targets: List[Tuple[int, int]]) -> bool:
        """Check if waypoint blocks all paths from sources to targets."""
        # Create modified screenshot with waypoint as walls
        modified_screenshot = screenshot.copy()
        x, y, radius = waypoint['x'], waypoint['y'], waypoint['radius']
        
        self._mark_waypoint_as_walls(modified_screenshot, waypoint)
        
        # Check if connectivity is broken
        return not self._check_connectivity(modified_screenshot, sources, targets)
    
    def _mark_waypoint_as_walls(self, screenshot: np.ndarray, waypoint: Dict[str, float]):
        """Mark waypoint area as walls in the screenshot."""
        x, y, radius = waypoint['x'], waypoint['y'], waypoint['radius']
        height, width = screenshot.shape
        
        min_px = max(0, int(x - radius - 1))
        max_px = min(width, int(x + radius + 2))
        min_py = max(0, int(y - radius - 1))
        max_py = min(height, int(y + radius + 2))
        
        for px in range(min_px, max_px):
            for py in range(min_py, max_py):
                if (px - x) ** 2 + (py - y) ** 2 <= radius ** 2:
                    screenshot[py, px] = 1  # Mark as wall
    
    def _get_waypoint_edge_pixels(self, waypoint: Dict[str, float]) -> List[Tuple[int, int]]:
        """Get pixels on the edge of waypoint circle for recursion target."""
        x, y, radius = waypoint['x'], waypoint['y'], waypoint['radius']
        edge_pixels = []
        
        # Sample points on circle edge
        num_samples = max(8, int(2 * math.pi * radius))
        for i in range(num_samples):
            angle = 2 * math.pi * i / num_samples
            edge_x = int(x + radius * math.cos(angle))
            edge_y = int(y + radius * math.sin(angle))
            edge_pixels.append((edge_x, edge_y))
        
        return edge_pixels
    
    def _find_pixels_by_type(self, screenshot: np.ndarray, pixel_type: int) -> List[Tuple[int, int]]:
        """Find all pixels of a given type."""
        positions = np.where(screenshot == pixel_type)
        return list(zip(positions[1], positions[0]))  # (x, y) format
    
    def _line_of_sight_clear(self, screenshot: np.ndarray, x1: int, y1: int, 
                           x2: int, y2: int) -> bool:
        """Check if line-of-sight is clear between two points using Bresenham's algorithm."""
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        x_inc = 1 if x1 < x2 else -1
        y_inc = 1 if y1 < y2 else -1
        error = dx - dy
        
        while True:
            # Check bounds and wall
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
    
    def _check_connectivity(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], 
                          targets: List[Tuple[int, int]]) -> bool:
        """Check if sources can reach targets."""
        if not sources or not targets:
            return False
        
        height, width = screenshot.shape
        visited = set()
        queue = deque(sources)
        visited.update(sources)
        
        while queue:
            x, y = queue.popleft()
            
            if (x, y) in targets:
                return True
            
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                if (nx < 0 or nx >= width or ny < 0 or ny >= height or 
                    (nx, ny) in visited or screenshot[ny, nx] == 1):
                    continue
                
                visited.add((nx, ny))
                queue.append((nx, ny))
        
        return False


if __name__ == "__main__":
    # Test the improved corner turning algorithm
    print("Testing improved corner turning algorithm...")
    
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    
    from waypoint_test_runner import create_synthetic_test_cases
    
    test_cases = create_synthetic_test_cases()
    generator = ImprovedCornerTurningGenerator(balloon_iterations=50)
    
    for test_name, screenshot in test_cases:
        print(f"\nTesting on {test_name}:")
        
        import time
        start_time = time.time()
        waypoints = generator.generate_waypoints(screenshot)
        duration = time.time() - start_time
        
        score = improved_score_waypoint_list(screenshot, waypoints, penalize_skippable=True)
        print(f"  Generated {len(waypoints)} waypoints in {duration:.3f}s")
        print(f"  Score: {score:.2f}")
        
        if waypoints:
            for i, wp in enumerate(waypoints):
                print(f"    WP{i+1}: ({wp['x']:.1f}, {wp['y']:.1f}) r={wp['radius']:.1f}")
    
    print("\nImproved corner turning test completed!")