#!/usr/bin/env python3.13
"""
Standalone test for corner turning variants showing different developer interpretations.
"""

import numpy as np
import math
import random
from typing import List, Dict, Tuple, Optional


def pixel_to_world(x: float, y: float, screenshot_shape: Tuple[int, int]) -> Tuple[float, float]:
    """Simple pixel to world coordinate conversion."""
    # Assume screenshot represents an 800x600 world
    world_x = (x / screenshot_shape[1]) * 800
    world_y = (y / screenshot_shape[0]) * 600
    return world_x, world_y


def world_to_pixel(world_x: float, world_y: float, screenshot_shape: Tuple[int, int]) -> Tuple[float, float]:
    """Simple world to pixel coordinate conversion."""
    x = (world_x / 800) * screenshot_shape[1]
    y = (world_y / 600) * screenshot_shape[0]
    return x, y


class SimpleCornerTurning:
    """Base class for corner turning algorithm variants."""
    
    def __init__(self, max_waypoints: int = 10):
        self.max_waypoints = max_waypoints
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints for the given screenshot."""
        return []
    
    def _find_sources_and_sinks(self, screenshot: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Find source (3) and sink (4) pixels."""
        sources = list(zip(*np.where(screenshot == 3)))
        sinks = list(zip(*np.where(screenshot == 4)))
        return sources, sinks
    
    def _is_wall(self, screenshot: np.ndarray, y: int, x: int) -> bool:
        """Check if a pixel is a wall (1) or out of bounds."""
        if y < 0 or y >= screenshot.shape[0] or x < 0 or x >= screenshot.shape[1]:
            return True
        return screenshot[y, x] == 1


class StudentImplementationCornerTurning(SimpleCornerTurning):
    """
    THE STRUGGLING STUDENT
    
    Interpretation: Misunderstood the assignment, made classic beginner mistakes
    """
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        sources, sinks = self._find_sources_and_sinks(screenshot)
        if not sources or not sinks:
            return []
        
        waypoints = []
        
        # Student mistake: hardcoded to always generate exactly 3 waypoints
        for i in range(3):
            # Student mistake: random waypoint placement instead of corner detection
            x = random.randint(10, screenshot.shape[1] - 10)
            y = random.randint(10, screenshot.shape[0] - 10)
            
            # Student mistake: radius is just random
            radius = random.uniform(1, 8)
            
            # Student mistake: doesn't check if waypoint is valid
            world_x, world_y = pixel_to_world(x, y, screenshot.shape)
            
            waypoint = {
                'x': world_x,
                'y': world_y, 
                'radius': radius * (800 / screenshot.shape[1])
            }
            waypoints.append(waypoint)
            
        return waypoints


class LazyImplementationCornerTurning(SimpleCornerTurning):
    """
    THE LAZY PROGRAMMER
    
    Interpretation: Takes shortcuts, minimal effort solutions
    """
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        sources, sinks = self._find_sources_and_sinks(screenshot)
        if not sources or not sinks:
            return []
        
        # Lazy: just put waypoint in middle of first source and first sink
        source = sources[0]
        sink = sinks[0] 
        
        mid_x = (source[1] + sink[1]) // 2
        mid_y = (source[0] + sink[0]) // 2
        
        # Lazy: try a few fixed radii, pick first that doesn't obviously break
        for radius in [2.0, 5.0, 10.0]:
            if not self._is_wall(screenshot, mid_y, mid_x):
                world_x, world_y = pixel_to_world(mid_x, mid_y, screenshot.shape)
                return [{
                    'x': world_x,
                    'y': world_y,
                    'radius': radius * (800 / screenshot.shape[1])
                }]
        
        return []  # Lazy: give up if simple approach doesn't work


class BuggyImplementationCornerTurning(SimpleCornerTurning):
    """
    THE BUGGY IMPLEMENTATION
    
    Interpretation: Tried to do it right but made several critical errors
    """
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        sources, sinks = self._find_sources_and_sinks(screenshot)
        if not sources or not sinks:
            return []
        
        waypoints = []
        
        # Bug: off-by-one error in loop limit
        for i in range(self.max_waypoints + 1):  # Should be max_waypoints, not +1
            
            # Bug: coordinate confusion (x and y swapped)
            corner = self._find_corner_buggy(screenshot, sources, sinks)
            if corner is None:
                break
                
            # Bug: coordinate system confusion in conversion
            world_x, world_y = pixel_to_world(corner[0], corner[1], screenshot.shape)  # Swapped!
            
            waypoint = {
                'x': world_x,
                'y': world_y,
                'radius': 3.0 * (800 / screenshot.shape[1])
            }
            waypoints.append(waypoint)
            
            # Bug: potential infinite loop if corner detection always returns same point
            if len(waypoints) > 20:  # Emergency break to prevent infinite loop
                break
                
        return waypoints
    
    def _find_corner_buggy(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find corner with several bugs."""
        
        for source in sources:
            for sink in sinks:
                distance = int(math.sqrt((sink[0] - source[0])**2 + (sink[1] - source[1])**2))
                
                for step in range(0, min(distance + 1, 10)):  # Limit to prevent hang
                    # Bug: integer division instead of float
                    t = step // max(distance, 1)  # Should be step / distance
                    
                    x = int(source[1] * (1-t) + sink[1] * t)
                    y = int(source[0] * (1-t) + sink[0] * t)
                    
                    # Bug: boundary check is wrong but return something anyway
                    if 0 < x < screenshot.shape[1] and 0 < y < screenshot.shape[0]:
                        return (y, x)
                        
        return sources[0] if sources else None  # Fallback


class OverengineeredCornerTurning(SimpleCornerTurning):
    """
    THE OVERENGINEER
    
    Interpretation: Makes everything unnecessarily complex
    """
    
    def __init__(self, max_waypoints: int = 10):
        super().__init__(max_waypoints)
        # Overengineered: way too many configuration parameters
        self.pathfinding_algorithms = ['astar', 'dijkstra', 'rrt', 'genetic', 'neural']
        self.expansion_strategies = ['ml_based', 'physics', 'optimization', 'heuristic']
        self.performance_metrics = {}
        
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        sources, sinks = self._find_sources_and_sinks(screenshot)
        if not sources or not sinks:
            return []
        
        waypoints = []
        
        # Overengineered: use ensemble of methods and vote
        for waypoint_index in range(min(self.max_waypoints, 3)):  # Limit for demo
            candidate_waypoints = []
            
            for strategy in self.expansion_strategies[:2]:  # Limit for demo
                try:
                    wp = self._generate_waypoint_with_strategy(screenshot, sources, sinks, strategy)
                    if wp:
                        candidate_waypoints.append(wp)
                except Exception:
                    pass  # Overengineered: silently ignore failures in ensemble
            
            if not candidate_waypoints:
                break
                
            # Return first candidate (simplified voting)
            waypoints.append(candidate_waypoints[0])
                
        return waypoints
    
    def _generate_waypoint_with_strategy(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]], strategy: str) -> Optional[Dict[str, float]]:
        """Generate waypoint using specific overengineered strategy."""
        
        if sources and sinks:
            # Overengineered: complex calculation for simple midpoint
            source_centroid = np.mean(sources, axis=0)
            sink_centroid = np.mean(sinks, axis=0)
            
            x = (source_centroid[1] + sink_centroid[1]) / 2 + random.normal(0, 2)
            y = (source_centroid[0] + sink_centroid[0]) / 2 + random.normal(0, 2)
            
            world_x, world_y = pixel_to_world(x, y, screenshot.shape)
            return {
                'x': world_x,
                'y': world_y,
                'radius': 4.0 * (800 / screenshot.shape[1])
            }
        return None


class HeuristicRulesCornerTurning(SimpleCornerTurning):
    """
    THE PRACTICAL HEURISTICS ENGINEER
    
    Interpretation: Values rules of thumb, prefers simple robust solutions
    """
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        sources, sinks = self._find_sources_and_sinks(screenshot)
        if not sources or not sinks:
            return []
        
        waypoints = []
        
        # Rule-based approach: place waypoint at narrowest passage
        for source in sources:
            for sink in sinks:
                # Simple rule: place waypoint 1/3 of the way from source to sink
                x = int(source[1] * 0.66 + sink[1] * 0.34)
                y = int(source[0] * 0.66 + sink[0] * 0.34)
                
                if not self._is_wall(screenshot, y, x):
                    world_x, world_y = pixel_to_world(x, y, screenshot.shape)
                    
                    waypoint = {
                        'x': world_x,
                        'y': world_y,
                        'radius': 3.0 * (800 / screenshot.shape[1])
                    }
                    waypoints.append(waypoint)
                    break
            break  # Only process first source-sink pair
                    
        return waypoints


def test_corner_turning_variants():
    """Test all corner turning variants on a simple synthetic case."""
    
    # Create a simple test case
    test_screenshot = np.zeros((20, 30), dtype=int)
    
    # Add walls to create an L-shaped corridor
    test_screenshot[0, :] = 1  # Top wall
    test_screenshot[-1, :] = 1  # Bottom wall
    test_screenshot[:, 0] = 1  # Left wall
    test_screenshot[:, -1] = 1  # Right wall
    test_screenshot[10, 5:20] = 1  # Internal wall creating L-shape
    
    # Add source and sink
    test_screenshot[5, 2] = 3  # Source
    test_screenshot[15, 25] = 4  # Sink
    
    # Test all variants
    variants = [
        ("Student Implementation", StudentImplementationCornerTurning()),
        ("Lazy Implementation", LazyImplementationCornerTurning()),
        ("Buggy Implementation", BuggyImplementationCornerTurning()),
        ("Overengineered", OverengineeredCornerTurning()),
        ("Heuristic Rules", HeuristicRulesCornerTurning())
    ]
    
    print("Testing Corner Turning Algorithm Variants")
    print("=" * 60)
    print("Different developer personalities interpreting the same specification")
    print()
    print(f"Test case: L-shaped corridor with source at (5,2) and sink at (15,25)")
    print()
    
    for name, variant in variants:
        print(f"🧑‍💻 {name}:")
        try:
            waypoints = variant.generate_waypoints(test_screenshot)
            print(f"   Generated {len(waypoints)} waypoints")
            for i, wp in enumerate(waypoints):
                print(f"   Waypoint {i+1}: x={wp['x']:.1f}, y={wp['y']:.1f}, r={wp['radius']:.1f}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
        print()


if __name__ == "__main__":
    test_corner_turning_variants()