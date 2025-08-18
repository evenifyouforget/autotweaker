"""
Quick versions of creative algorithms for demonstration and fast testing.

These are optimized for speed while maintaining the core algorithmic concepts.
"""

import numpy as np
import math
import random
from typing import List, Dict, Tuple, Optional
import time

try:
    from .waypoint_generation import WaypointGenerator
    from .improved_waypoint_scoring import improved_score_waypoint_list
except ImportError:
    from waypoint_generation import WaypointGenerator
    from improved_waypoint_scoring import improved_score_waypoint_list


class QuickGeneticGenerator(WaypointGenerator):
    """Fast genetic algorithm for demonstrations."""
    
    def __init__(self):
        super().__init__("QuickGenetic")
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Quick genetic algorithm with reduced parameters."""
        height, width = screenshot.shape
        
        # Find passable points
        passable = []
        for y in range(5, height-5, 4):  # Reduced sampling
            for x in range(5, width-5, 4):
                if screenshot[y, x] != 1:
                    passable.append((x, y))
        
        if len(passable) < 5:
            return []
        
        best_waypoints = []
        best_score = float('inf')
        
        # Quick evolution: only 20 attempts
        for _ in range(20):
            # Random waypoint configuration
            num_waypoints = random.randint(0, 3)  # Limit to 3 waypoints max
            waypoints = []
            
            if num_waypoints > 0:
                selected = random.sample(passable, min(num_waypoints, len(passable)))
                for x, y in selected:
                    radius = random.uniform(4.0, 12.0)
                    waypoints.append({'x': float(x), 'y': float(y), 'radius': radius})
            
            # Quick scoring
            score = improved_score_waypoint_list(screenshot, waypoints, 
                                               feature_flags={'use_ant_simulation': False,
                                                            'check_local_valleys_proper': False})
            
            if score < best_score:
                best_score = score
                best_waypoints = waypoints
        
        return best_waypoints


class QuickFlowFieldGenerator(WaypointGenerator):
    """Fast flow field generator."""
    
    def __init__(self):
        super().__init__("QuickFlowField")
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Quick flow field analysis."""
        height, width = screenshot.shape
        
        # Find sources and sinks
        sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
        sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
        
        if not sources or not sinks:
            return []
        
        waypoints = []
        
        # Simple approach: find midpoints with high gradient
        for y in range(10, height-10, 8):  # Reduced sampling
            for x in range(10, width-10, 8):
                if screenshot[y, x] == 1:  # Skip walls
                    continue
                
                # Calculate simple gradient
                grad_x = float(screenshot[y, x+1]) - float(screenshot[y, x-1])
                grad_y = float(screenshot[y+1, x]) - float(screenshot[y-1, x])
                grad_mag = math.sqrt(grad_x**2 + grad_y**2)
                
                # Look for high gradient areas
                if grad_mag > 1.5:
                    radius = 6.0
                    waypoints.append({'x': float(x), 'y': float(y), 'radius': radius})
        
        # Return top 3 candidates
        return waypoints[:3]


class QuickSwarmGenerator(WaypointGenerator):
    """Fast swarm intelligence generator."""
    
    def __init__(self):
        super().__init__("QuickSwarm")
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Quick swarm optimization."""
        height, width = screenshot.shape
        
        # Find bounds
        passable_y, passable_x = np.where(screenshot != 1)
        if len(passable_x) == 0:
            return []
        
        min_x, max_x = int(np.min(passable_x)) + 5, int(np.max(passable_x)) - 5
        min_y, max_y = int(np.min(passable_y)) + 5, int(np.max(passable_y)) - 5
        
        best_waypoints = []
        best_score = float('inf')
        
        # Quick swarm: 15 particles, 10 iterations
        for _ in range(15):
            # Random particle
            num_waypoints = random.randint(0, 2)
            waypoints = []
            
            for _ in range(num_waypoints):
                x = random.uniform(min_x, max_x)
                y = random.uniform(min_y, max_y)
                radius = random.uniform(4.0, 10.0)
                waypoints.append({'x': x, 'y': y, 'radius': radius})
            
            # Quick evaluation
            score = improved_score_waypoint_list(screenshot, waypoints,
                                               feature_flags={'use_ant_simulation': False})
            
            if score < best_score:
                best_score = score
                best_waypoints = waypoints
        
        return best_waypoints


class QuickAdaptiveGenerator(WaypointGenerator):
    """Fast adaptive random generator."""
    
    def __init__(self):
        super().__init__("QuickAdaptive")
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Quick adaptive sampling."""
        height, width = screenshot.shape
        
        best_waypoints = []
        best_score = float('inf')
        
        # 25 quick attempts
        for _ in range(25):
            # Random configuration
            num_waypoints = random.randint(0, 3)
            waypoints = []
            
            for _ in range(num_waypoints):
                # Sample random position
                attempts = 0
                while attempts < 10:
                    x = random.randint(5, width-5)
                    y = random.randint(5, height-5)
                    if screenshot[y, x] != 1:  # Not a wall
                        radius = random.uniform(3.0, 10.0)
                        waypoints.append({'x': float(x), 'y': float(y), 'radius': radius})
                        break
                    attempts += 1
            
            # Quick evaluation
            score = improved_score_waypoint_list(screenshot, waypoints,
                                               feature_flags={'use_ant_simulation': False})
            
            if score < best_score:
                best_score = score
                best_waypoints = waypoints
        
        return best_waypoints


def create_quick_creative_tournament():
    """Create a tournament with quick creative algorithms for fast demonstration."""
    try:
        from .waypoint_generation import WaypointTournament, NullGenerator, CornerTurningGenerator
        from .improved_corner_turning import ImprovedCornerTurningGenerator
    except ImportError:
        from waypoint_generation import WaypointTournament, NullGenerator, CornerTurningGenerator
        from improved_corner_turning import ImprovedCornerTurningGenerator
    
    tournament = WaypointTournament()
    
    # Add basic algorithms
    tournament.add_generator(NullGenerator())
    tournament.add_generator(CornerTurningGenerator())
    
    # Add quick creative algorithms
    tournament.add_generator(QuickGeneticGenerator())
    tournament.add_generator(QuickFlowFieldGenerator())
    tournament.add_generator(QuickSwarmGenerator())
    tournament.add_generator(QuickAdaptiveGenerator())
    
    # Add improved corner turning
    try:
        tournament.add_generator(ImprovedCornerTurningGenerator(balloon_iterations=20, max_waypoints=5))
    except:
        pass  # Skip if not available
    
    return tournament


if __name__ == "__main__":
    # Test quick creative algorithms
    print("Testing quick creative algorithms...")
    
    from waypoint_test_runner import create_synthetic_test_cases
    
    test_cases = create_synthetic_test_cases()
    tournament = create_quick_creative_tournament()
    
    print(f"Created tournament with {len(tournament.generators)} algorithms")
    
    # Test on L-shape case
    test_name, screenshot = test_cases[1]
    print(f"\nQuick test on {test_name}:")
    
    for generator in tournament.generators:
        try:
            start_time = time.time()
            waypoints = generator.generate_waypoints(screenshot)
            duration = time.time() - start_time
            
            score = improved_score_waypoint_list(screenshot, waypoints, 
                                               feature_flags={'use_ant_simulation': False})
            
            print(f"  {generator.name:15} | {len(waypoints):2d} waypoints | {duration:5.3f}s | Score: {score:8.2f}")
            
        except Exception as e:
            print(f"  {generator.name:15} | ERROR: {e}")
    
    print("\nQuick creative algorithms test completed!")