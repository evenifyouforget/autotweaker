"""
Waypoint generation algorithms and tournament system.

This module provides a framework for different waypoint generation algorithms
and a tournament system to evaluate and compare their performance.
"""

import numpy as np
import math
import time
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Optional, Type, Any
from collections import defaultdict, deque
import random

try:
    from .waypoint_scoring import score_waypoint_list, check_waypoint_non_skippability
except ImportError:
    from waypoint_scoring import score_waypoint_list, check_waypoint_non_skippability


class WaypointGenerator(ABC):
    """Base class for waypoint generation algorithms."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """
        Generate waypoints for a given screenshot.
        
        Args:
            screenshot: 2D array where 1=wall, 3=source, 4=sink, 0=passable
        
        Returns:
            List of waypoint dicts with keys 'x', 'y', 'radius' (in pixel coordinates)
        """
        pass
    
    def __str__(self) -> str:
        return self.name


class NullGenerator(WaypointGenerator):
    """Baseline generator that returns an empty waypoint list."""
    
    def __init__(self):
        super().__init__("Null")
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Return empty waypoint list."""
        return []


class CornerTurningGenerator(WaypointGenerator):
    """
    Recursive corner-turning waypoint generation algorithm.
    
    This algorithm works backwards from sink to source, placing waypoints
    at corners where direct line-of-sight is lost.
    """
    
    def __init__(self, max_iterations: int = 100, max_waypoints: int = 20, 
                 balloon_iterations: int = 50):
        super().__init__("CornerTurning")
        self.max_iterations = max_iterations
        self.max_waypoints = max_waypoints
        self.balloon_iterations = balloon_iterations
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using corner-turning algorithm."""
        sources = self._find_pixels_by_type(screenshot, 3)
        sinks = self._find_pixels_by_type(screenshot, 4)
        
        if not sources or not sinks:
            return []
        
        # Start recursive algorithm
        waypoints = self._corner_algorithm(screenshot, sources, sinks)
        return waypoints[:self.max_waypoints]  # Limit waypoint count
    
    def _corner_algorithm(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], 
                         targets: List[Tuple[int, int]]) -> List[Dict[str, float]]:
        """Recursive corner-turning algorithm."""
        
        # Check if direct line-of-sight exists from all sources to any target
        if self._has_direct_line_of_sight(screenshot, sources, targets):
            return []  # No waypoint needed
        
        # Find a corner by pathfinding and tracing back
        path = self._find_path(screenshot, sources, targets)
        if not path:
            return []  # No path exists
        
        # Find first obscured point along the path from targets
        corner_point = self._find_first_obscured_point(screenshot, path, targets)
        if corner_point is None:
            return []
        
        # Create candidate waypoint at corner
        waypoint = self._create_balloon_waypoint(screenshot, corner_point)
        if waypoint is None:
            return []
        
        # Check if this waypoint makes paths non-skippable
        if not self._is_waypoint_blocking(screenshot, waypoint, sources, targets):
            return []
        
        # Recursively find waypoints from sources to this waypoint
        recursive_waypoints = self._corner_algorithm(screenshot, sources, [(int(waypoint['x']), int(waypoint['y']))])
        
        return recursive_waypoints + [waypoint]
    
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
    
    def _find_path(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], 
                  targets: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Find a path from sources to targets using BFS."""
        if not sources or not targets:
            return []
        
        height, width = screenshot.shape
        visited = {}
        queue = deque()
        
        # Start from all targets (working backwards)
        for target in targets:
            queue.append(target)
            visited[target] = None
        
        # BFS
        while queue:
            x, y = queue.popleft()
            
            # Check if we reached a source
            if (x, y) in sources:
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
    
    def _find_first_obscured_point(self, screenshot: np.ndarray, path: List[Tuple[int, int]], 
                                  targets: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find first point on path that doesn't have line-of-sight to any target."""
        for point in reversed(path):  # Work backwards from source to target
            if not self._has_direct_line_of_sight(screenshot, [point], targets):
                return point
        return None
    
    def _create_balloon_waypoint(self, screenshot: np.ndarray, 
                               center: Tuple[int, int]) -> Optional[Dict[str, float]]:
        """Create a waypoint by expanding like a balloon from the center point."""
        x, y = center
        
        # Start with small radius
        radius = 5.0
        best_radius = radius
        
        for _ in range(self.balloon_iterations):
            # Check if waypoint at this radius is valid
            if self._is_waypoint_valid(screenshot, x, y, radius):
                best_radius = radius
                # Try to expand
                radius += 2.0
            else:
                # Can't expand further
                break
        
        if best_radius < 1.0:
            return None
        
        return {'x': float(x), 'y': float(y), 'radius': best_radius}
    
    def _is_waypoint_valid(self, screenshot: np.ndarray, x: float, y: float, radius: float) -> bool:
        """Check if a waypoint doesn't intersect with walls."""
        height, width = screenshot.shape
        
        # Check circle doesn't go out of bounds or hit walls
        min_px = max(0, int(x - radius - 1))
        max_px = min(width, int(x + radius + 2))
        min_py = max(0, int(y - radius - 1))
        max_py = min(height, int(y + radius + 2))
        
        for px in range(min_px, max_px):
            for py in range(min_py, max_py):
                if (px - x) ** 2 + (py - y) ** 2 <= radius ** 2:
                    if screenshot[py, px] == 1:  # Wall
                        return False
        
        return True
    
    def _is_waypoint_blocking(self, screenshot: np.ndarray, waypoint: Dict[str, float],
                            sources: List[Tuple[int, int]], targets: List[Tuple[int, int]]) -> bool:
        """Check if waypoint blocks all paths from sources to targets."""
        # Create modified screenshot with waypoint as walls
        modified_screenshot = screenshot.copy()
        x, y, radius = waypoint['x'], waypoint['y'], waypoint['radius']
        
        height, width = screenshot.shape
        min_px = max(0, int(x - radius - 1))
        max_px = min(width, int(x + radius + 2))
        min_py = max(0, int(y - radius - 1))
        max_py = min(height, int(y + radius + 2))
        
        for px in range(min_px, max_px):
            for py in range(min_py, max_py):
                if (px - x) ** 2 + (py - y) ** 2 <= radius ** 2:
                    modified_screenshot[py, px] = 1  # Mark as wall
        
        # Check if connectivity is broken
        return not self._check_connectivity(modified_screenshot, sources, targets)
    
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


class WaypointTournament:
    """Tournament system for evaluating waypoint generation algorithms."""
    
    def __init__(self, feature_flags: Optional[Dict[str, bool]] = None):
        self.generators: List[WaypointGenerator] = []
        self.feature_flags = feature_flags or {}
        self.results: Dict[str, Dict[str, Any]] = {}
    
    def add_generator(self, generator: WaypointGenerator):
        """Add a generator to the tournament."""
        self.generators.append(generator)
    
    def add_generator_class(self, generator_class: Type[WaypointGenerator], *args, **kwargs):
        """Add a generator class to the tournament."""
        generator = generator_class(*args, **kwargs)
        self.add_generator(generator)
    
    def run_tournament(self, test_cases: List[Tuple[str, np.ndarray]], 
                      verbose: bool = True) -> Dict[str, Dict[str, Any]]:
        """
        Run the tournament on all test cases.
        
        Args:
            test_cases: List of (test_name, screenshot) tuples
            verbose: Whether to print progress information
        
        Returns:
            Dict mapping generator names to their results
        """
        if verbose:
            print(f"Running tournament with {len(self.generators)} generators on {len(test_cases)} test cases...")
        
        # Initialize results
        for generator in self.generators:
            self.results[generator.name] = {
                'scores': [],
                'times': [],
                'waypoint_counts': [],
                'skippable_count': 0,
                'error_count': 0,
                'total_score': 0.0,
                'avg_time': 0.0,
                'avg_waypoints': 0.0
            }
        
        # Run each generator on each test case
        for test_name, screenshot in test_cases:
            if verbose:
                print(f"\nTesting on {test_name}:")
            
            for generator in self.generators:
                try:
                    # Time the generation
                    start_time = time.time()
                    waypoints = generator.generate_waypoints(screenshot)
                    generation_time = time.time() - start_time
                    
                    # Score the waypoints
                    score = score_waypoint_list(screenshot, waypoints, 
                                              penalize_skippable=True, 
                                              feature_flags=self.feature_flags)
                    
                    # Check if waypoints are skippable
                    is_skippable = not check_waypoint_non_skippability(screenshot, waypoints)
                    
                    # Record results
                    results = self.results[generator.name]
                    results['scores'].append(score)
                    results['times'].append(generation_time)
                    results['waypoint_counts'].append(len(waypoints))
                    if is_skippable:
                        results['skippable_count'] += 1
                    
                    if verbose:
                        status = "SKIP" if is_skippable else "OK"
                        print(f"  {generator.name:15} | Score: {score:8.2f} | "
                              f"Time: {generation_time:6.3f}s | Waypoints: {len(waypoints):2d} | {status}")
                
                except Exception as e:
                    self.results[generator.name]['error_count'] += 1
                    if verbose:
                        print(f"  {generator.name:15} | ERROR: {str(e)}")
        
        # Calculate aggregate statistics
        self._calculate_final_rankings()
        
        return self.results
    
    def _calculate_final_rankings(self):
        """Calculate final rankings and statistics for all generators."""
        # Normalize scores across test cases for ranking
        all_scores = []
        for generator_name, results in self.results.items():
            all_scores.extend(results['scores'])
        
        if not all_scores:
            return
        
        # Calculate statistics for each generator
        for generator_name, results in self.results.items():
            scores = results['scores']
            times = results['times']
            waypoint_counts = results['waypoint_counts']
            
            if scores:
                results['total_score'] = sum(scores)
                results['avg_score'] = sum(scores) / len(scores)
                results['median_score'] = sorted(scores)[len(scores) // 2]
            
            if times:
                results['avg_time'] = sum(times) / len(times)
                results['max_time'] = max(times)
            
            if waypoint_counts:
                results['avg_waypoints'] = sum(waypoint_counts) / len(waypoint_counts)
    
    def print_final_rankings(self):
        """Print final tournament rankings."""
        if not self.results:
            print("No results to display.")
            return
        
        print("\n" + "="*80)
        print("FINAL TOURNAMENT RANKINGS")
        print("="*80)
        
        # Sort by total score (lower is better)
        sorted_results = sorted(self.results.items(), key=lambda x: x[1]['total_score'])
        
        print(f"{'Rank':<4} {'Generator':<15} {'Total Score':<12} {'Avg Score':<10} "
              f"{'Skippable':<9} {'Errors':<7} {'Avg Time':<9} {'Avg WP':<7}")
        print("-" * 80)
        
        for rank, (name, results) in enumerate(sorted_results, 1):
            print(f"{rank:<4} {name:<15} {results['total_score']:<12.2f} "
                  f"{results.get('avg_score', 0):<10.2f} "
                  f"{results['skippable_count']:<9} {results['error_count']:<7} "
                  f"{results['avg_time']:<9.3f} {results.get('avg_waypoints', 0):<7.1f}")
        
        print("="*80)


def get_all_generator_classes() -> List[Type[WaypointGenerator]]:
    """Get all available waypoint generator classes."""
    return [NullGenerator, CornerTurningGenerator]


def create_default_tournament() -> WaypointTournament:
    """Create a tournament with default generators and settings."""
    tournament = WaypointTournament()
    
    # Add all available generators
    tournament.add_generator(NullGenerator())
    tournament.add_generator(CornerTurningGenerator())
    
    # Add some variations
    tournament.add_generator(CornerTurningGenerator(max_waypoints=10, balloon_iterations=30))
    
    return tournament