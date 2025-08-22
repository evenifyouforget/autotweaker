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
    
    def __init__(self, name: Optional[str] = None):
        # Auto-derive name from class name if not provided
        self.name = name or self.__class__.__name__.replace('Generator', '')
    
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
        """Check if a waypoint is valid (waypoints can overlap walls)."""
        height, width = screenshot.shape
        
        # Only check bounds - waypoints are allowed to overlap with walls
        if x - radius < 0 or x + radius >= width:
            return False
        if y - radius < 0 or y + radius >= height:
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


# Import the new high-performance tournament implementation
from multithreaded_tournament import WaypointTournament, TournamentConfig


def get_all_generators() -> List[WaypointGenerator]:
    """Get all available waypoint generator classes using reflection."""
    def get_all_subclasses(cls):
        """Recursively get all subclasses of a class."""
        subclasses = set(cls.__subclasses__())
        for subclass in list(subclasses):
            subclasses.update(get_all_subclasses(subclass))
        return subclasses
    
    # Import all modules to make sure subclasses are registered
    import importlib
    import os
    import sys
    
    # Get current directory 
    current_dir = os.path.dirname(__file__)
    
    # Import all Python files in current directory to register subclasses
    for filename in os.listdir(current_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            module_name = filename[:-3]
            try:
                if __package__:
                    importlib.import_module(f'.{module_name}', package=__package__)
                else:
                    importlib.import_module(module_name)
            except ImportError as e:
                # Skip modules that can't be imported
                continue
    
    # Get all subclasses
    all_subclasses = get_all_subclasses(WaypointGenerator)
    
    # Filter out abstract classes and return concrete instances
    concrete_instances = []
    for cls in all_subclasses:
        try:
            # Try to instantiate to check if it's concrete
            instance = cls()
            concrete_instances.append(instance)
        except (TypeError, NotImplementedError):
            # Skip abstract classes or classes that can't be instantiated
            continue
        except Exception:
            # Skip classes with other instantiation issues
            continue
    
    return concrete_instances

def get_all_generator_classes() -> List[Type[WaypointGenerator]]:
    """Get all available waypoint generator classes (for compatibility)."""
    instances = get_all_generators()
    return [instance.__class__ for instance in instances]

def get_generator_by_name(name: str) -> Optional[WaypointGenerator]:
    """Get a generator instance by name."""
    for generator in get_all_generators():
        if generator.name == name:
            return generator
    return None

def create_generator(name: str) -> WaypointGenerator:
    """Create a generator instance by name."""
    generator = get_generator_by_name(name)
    if generator is None:
        available_names = [g.name for g in get_all_generators()]
        raise ValueError(f"Unknown generator: {name}. Available: {available_names}")
    return generator.__class__()


def create_default_tournament(max_workers: Optional[int] = 1):
    """Create a tournament with default generators and settings.
    
    Args:
        max_workers: Number of workers (1 = single-threaded for compatibility)
    """
    # Import the new tournament implementation
    from multithreaded_tournament import create_default_tournament as create_new_tournament
    
    # Use single-threaded by default for backward compatibility
    return create_new_tournament(max_workers=max_workers)