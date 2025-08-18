"""
Advanced waypoint generation algorithms using image processing and graph theory.

This module implements more sophisticated waypoint generation algorithms that
leverage computer vision and graph algorithms for better performance.
"""

import numpy as np
import math
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
import networkx as nx
from scipy import ndimage
from skimage import morphology, measure, segmentation
from skimage.feature import corner_harris, corner_peaks
import matplotlib.pyplot as plt

try:
    from .waypoint_generation import WaypointGenerator
    from .waypoint_scoring import check_waypoint_non_skippability
except ImportError:
    from waypoint_generation import WaypointGenerator
    from waypoint_scoring import check_waypoint_non_skippability


class MedialAxisGenerator(WaypointGenerator):
    """
    Generate waypoints along the medial axis (skeleton) of passable areas.
    
    This algorithm finds the skeleton of the passable area and places waypoints
    at critical points along this skeleton, particularly at junctions and turns.
    """
    
    def __init__(self, min_waypoint_distance: float = 20.0, max_waypoints: int = 15):
        super().__init__("MedialAxis")
        self.min_waypoint_distance = min_waypoint_distance
        self.max_waypoints = max_waypoints
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints along the medial axis."""
        # Find passable areas (not walls)
        passable = (screenshot != 1).astype(np.uint8)
        
        # Compute medial axis (skeleton)
        skeleton = morphology.skeletonize(passable)
        
        # Find critical points on skeleton
        waypoint_candidates = self._find_critical_points(skeleton, screenshot)
        
        # Filter and order waypoints
        waypoints = self._select_optimal_waypoints(waypoint_candidates, screenshot)
        
        return waypoints[:self.max_waypoints]
    
    def _find_critical_points(self, skeleton: np.ndarray, screenshot: np.ndarray) -> List[Tuple[int, int]]:
        """Find critical points (junctions, endpoints) on the skeleton."""
        critical_points = []
        
        # Find skeleton pixels
        skeleton_pixels = np.where(skeleton)
        
        for y, x in zip(skeleton_pixels[0], skeleton_pixels[1]):
            # Count neighboring skeleton pixels
            neighbors = 0
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and
                        skeleton[ny, nx]):
                        neighbors += 1
            
            # Critical points: junctions (3+ neighbors) or endpoints (1 neighbor)
            if neighbors == 1 or neighbors >= 3:
                critical_points.append((x, y))
        
        return critical_points
    
    def _select_optimal_waypoints(self, candidates: List[Tuple[int, int]], 
                                screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Select and order waypoints from candidates."""
        if not candidates:
            return []
        
        sources = self._find_pixels_by_type(screenshot, 3)
        sinks = self._find_pixels_by_type(screenshot, 4)
        
        if not sources or not sinks:
            return []
        
        # Sort candidates by distance from sources (closest first)
        source_center = (sum(x for x, y in sources) / len(sources),
                        sum(y for x, y in sources) / len(sources))
        
        candidates_with_distance = []
        for x, y in candidates:
            dist = math.sqrt((x - source_center[0])**2 + (y - source_center[1])**2)
            candidates_with_distance.append((dist, x, y))
        
        candidates_with_distance.sort()
        
        # Select waypoints with minimum distance constraint
        selected_waypoints = []
        for _, x, y in candidates_with_distance:
            # Check minimum distance from existing waypoints
            too_close = False
            for wp in selected_waypoints:
                dist = math.sqrt((x - wp['x'])**2 + (y - wp['y'])**2)
                if dist < self.min_waypoint_distance:
                    too_close = True
                    break
            
            if not too_close:
                # Estimate reasonable radius
                radius = self._estimate_corridor_width(screenshot, x, y)
                selected_waypoints.append({'x': float(x), 'y': float(y), 'radius': radius})
        
        return selected_waypoints
    
    def _find_pixels_by_type(self, screenshot: np.ndarray, pixel_type: int) -> List[Tuple[int, int]]:
        """Find all pixels of a given type."""
        positions = np.where(screenshot == pixel_type)
        return list(zip(positions[1], positions[0]))  # (x, y) format
    
    def _estimate_corridor_width(self, screenshot: np.ndarray, x: int, y: int) -> float:
        """Estimate the width of the corridor at a given point."""
        # Find distance to nearest wall in multiple directions
        distances = []
        
        for angle in np.linspace(0, 2*math.pi, 16):
            dx = math.cos(angle)
            dy = math.sin(angle)
            
            for dist in range(1, 50):
                test_x = int(x + dx * dist)
                test_y = int(y + dy * dist)
                
                if (test_x < 0 or test_x >= screenshot.shape[1] or 
                    test_y < 0 or test_y >= screenshot.shape[0] or
                    screenshot[test_y, test_x] == 1):  # Hit wall
                    distances.append(dist)
                    break
        
        if distances:
            return min(distances) * 0.8  # Use 80% of minimum distance
        return 5.0  # Default radius


class VoronoiGenerator(WaypointGenerator):
    """
    Generate waypoints using Voronoi diagram of walls.
    
    This algorithm creates a Voronoi diagram where walls are the seed points,
    then places waypoints along the Voronoi edges that are furthest from walls.
    """
    
    def __init__(self, min_distance_from_wall: float = 10.0, max_waypoints: int = 12):
        super().__init__("Voronoi")
        self.min_distance_from_wall = min_distance_from_wall
        self.max_waypoints = max_waypoints
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using Voronoi diagram approach."""
        # Compute distance transform from walls
        walls = (screenshot == 1)
        distance_transform = ndimage.distance_transform_edt(~walls)
        
        # Find local maxima in distance transform (points furthest from walls)
        local_maxima = self._find_local_maxima(distance_transform)
        
        # Filter maxima by minimum distance from walls
        filtered_maxima = []
        for y, x in local_maxima:
            if distance_transform[y, x] >= self.min_distance_from_wall:
                filtered_maxima.append((x, y, distance_transform[y, x]))
        
        # Sort by distance from walls (furthest first)
        filtered_maxima.sort(key=lambda p: p[2], reverse=True)
        
        # Convert to waypoints and ensure path connectivity
        waypoints = self._create_connected_waypoints(filtered_maxima, screenshot)
        
        return waypoints[:self.max_waypoints]
    
    def _find_local_maxima(self, distance_transform: np.ndarray) -> List[Tuple[int, int]]:
        """Find local maxima in the distance transform."""
        # Use a simple peak finding approach
        maxima = []
        h, w = distance_transform.shape
        
        for y in range(1, h-1):
            for x in range(1, w-1):
                center = distance_transform[y, x]
                if center <= 2:  # Skip points too close to walls
                    continue
                
                # Check if this is a local maximum
                is_maximum = True
                for dy in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dy == 0 and dx == 0:
                            continue
                        if distance_transform[y + dy, x + dx] > center:
                            is_maximum = False
                            break
                    if not is_maximum:
                        break
                
                if is_maximum:
                    maxima.append((y, x))
        
        return maxima
    
    def _create_connected_waypoints(self, maxima: List[Tuple[int, int, float]], 
                                  screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Create waypoints ensuring path connectivity."""
        if not maxima:
            return []
        
        sources = self._find_pixels_by_type(screenshot, 3)
        sinks = self._find_pixels_by_type(screenshot, 4)
        
        if not sources or not sinks:
            return []
        
        # Find path from sources to sinks
        path = self._find_path(screenshot, sources, sinks)
        if not path:
            return []
        
        # Select maxima that are close to the path
        waypoints = []
        for x, y, dist_from_wall in maxima:
            # Find closest point on path
            min_path_distance = float('inf')
            for path_x, path_y in path:
                path_dist = math.sqrt((x - path_x)**2 + (y - path_y)**2)
                min_path_distance = min(min_path_distance, path_dist)
            
            # Include if reasonably close to path
            if min_path_distance <= 20:
                radius = min(dist_from_wall * 0.7, 15.0)  # Limit maximum radius
                waypoints.append({'x': float(x), 'y': float(y), 'radius': radius})
        
        return waypoints
    
    def _find_pixels_by_type(self, screenshot: np.ndarray, pixel_type: int) -> List[Tuple[int, int]]:
        """Find all pixels of a given type."""
        positions = np.where(screenshot == pixel_type)
        return list(zip(positions[1], positions[0]))  # (x, y) format
    
    def _find_path(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], 
                  targets: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Find a path from sources to targets using BFS."""
        if not sources or not targets:
            return []
        
        height, width = screenshot.shape
        visited = {}
        queue = deque()
        
        # Start from sources
        for source in sources:
            queue.append(source)
            visited[source] = None
        
        # BFS
        while queue:
            x, y = queue.popleft()
            
            # Check if we reached a target
            if (x, y) in targets:
                # Reconstruct path
                path = []
                current = (x, y)
                while current is not None:
                    path.append(current)
                    current = visited[current]
                return path
            
            # Explore neighbors
            for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nx, ny = x + dx, y + dy
                
                # Check bounds
                if nx < 0 or nx >= width or ny < 0 or ny >= height:
                    continue
                
                # Check if already visited
                if (nx, ny) in visited:
                    continue
                
                # Check if passable
                if screenshot[ny, nx] == 1:
                    continue
                
                visited[(nx, ny)] = (x, y)
                queue.append((nx, ny))
        
        return []


class OptimizedSearchGenerator(WaypointGenerator):
    """
    Generate waypoints using simulated annealing optimization.
    
    This algorithm starts with random waypoints and optimizes their positions
    and radii using simulated annealing to minimize the scoring function.
    """
    
    def __init__(self, max_waypoints: int = 8, iterations: int = 1000, 
                 initial_temp: float = 100.0):
        super().__init__("OptimizedSearch")
        self.max_waypoints = max_waypoints
        self.iterations = iterations
        self.initial_temp = initial_temp
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using simulated annealing optimization."""
        # Import scoring function
        try:
            from .waypoint_scoring import score_waypoint_list
        except ImportError:
            from waypoint_scoring import score_waypoint_list
        
        # Find valid area for waypoint placement
        passable_points = self._find_passable_points(screenshot)
        if len(passable_points) < 10:
            return []
        
        # Try different numbers of waypoints
        best_waypoints = []
        best_score = float('inf')
        
        for num_waypoints in range(0, self.max_waypoints + 1):
            if num_waypoints == 0:
                candidate = []
            else:
                candidate = self._optimize_waypoints(screenshot, num_waypoints, 
                                                   passable_points, score_waypoint_list)
            
            score = score_waypoint_list(screenshot, candidate, penalize_skippable=True)
            if score < best_score:
                best_score = score
                best_waypoints = candidate
        
        return best_waypoints
    
    def _find_passable_points(self, screenshot: np.ndarray) -> List[Tuple[int, int]]:
        """Find points where waypoints can be placed."""
        passable = []
        height, width = screenshot.shape
        
        # Sample passable area
        for y in range(5, height - 5, 3):
            for x in range(5, width - 5, 3):
                if screenshot[y, x] != 1:  # Not a wall
                    passable.append((x, y))
        
        return passable
    
    def _optimize_waypoints(self, screenshot: np.ndarray, num_waypoints: int,
                          passable_points: List[Tuple[int, int]], score_func) -> List[Dict[str, float]]:
        """Optimize waypoint positions using simulated annealing."""
        import random
        
        if num_waypoints == 0:
            return []
        
        # Initialize random waypoints
        current_waypoints = []
        for _ in range(num_waypoints):
            x, y = random.choice(passable_points)
            radius = random.uniform(3.0, 15.0)
            current_waypoints.append({'x': float(x), 'y': float(y), 'radius': radius})
        
        current_score = score_func(screenshot, current_waypoints, penalize_skippable=True)
        best_waypoints = current_waypoints.copy()
        best_score = current_score
        
        temperature = self.initial_temp
        
        for iteration in range(self.iterations):
            # Create neighbor solution
            new_waypoints = self._create_neighbor_solution(current_waypoints, passable_points)
            new_score = score_func(screenshot, new_waypoints, penalize_skippable=True)
            
            # Accept or reject
            delta = new_score - current_score
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_waypoints = new_waypoints
                current_score = new_score
                
                if new_score < best_score:
                    best_waypoints = new_waypoints.copy()
                    best_score = new_score
            
            # Cool down
            temperature = self.initial_temp * (0.99 ** iteration)
        
        return best_waypoints
    
    def _create_neighbor_solution(self, waypoints: List[Dict[str, float]], 
                                passable_points: List[Tuple[int, int]]) -> List[Dict[str, float]]:
        """Create a neighbor solution by slightly modifying current waypoints."""
        import random
        
        new_waypoints = []
        for wp in waypoints:
            new_wp = wp.copy()
            
            # Randomly modify position or radius
            if random.random() < 0.7:  # Modify position
                dx = random.uniform(-10, 10)
                dy = random.uniform(-10, 10)
                new_wp['x'] = max(5, min(new_wp['x'] + dx, 195))  # Keep in bounds
                new_wp['y'] = max(5, min(new_wp['y'] + dy, 140))
            else:  # Modify radius
                dr = random.uniform(-3, 3)
                new_wp['radius'] = max(2.0, min(new_wp['radius'] + dr, 20.0))
            
            new_waypoints.append(new_wp)
        
        return new_waypoints


def create_advanced_tournament():
    """Create a tournament with advanced generators."""
    try:
        from .waypoint_generation import WaypointTournament, NullGenerator, CornerTurningGenerator
    except ImportError:
        from waypoint_generation import WaypointTournament, NullGenerator, CornerTurningGenerator
    
    tournament = WaypointTournament()
    
    # Add basic generators
    tournament.add_generator(NullGenerator())
    tournament.add_generator(CornerTurningGenerator())
    
    # Add advanced generators
    tournament.add_generator(MedialAxisGenerator())
    tournament.add_generator(VoronoiGenerator())
    tournament.add_generator(OptimizedSearchGenerator(max_waypoints=5, iterations=500))
    
    # Add variations
    tournament.add_generator(MedialAxisGenerator(min_waypoint_distance=15.0, max_waypoints=10))
    tournament.add_generator(VoronoiGenerator(min_distance_from_wall=8.0, max_waypoints=8))
    
    return tournament


if __name__ == "__main__":
    # Test advanced generators
    print("Testing advanced waypoint generators...")
    
    from waypoint_test_runner import create_synthetic_test_cases
    
    test_cases = create_synthetic_test_cases()
    tournament = create_advanced_tournament()
    
    results = tournament.run_tournament(test_cases, verbose=True)
    tournament.print_final_rankings()