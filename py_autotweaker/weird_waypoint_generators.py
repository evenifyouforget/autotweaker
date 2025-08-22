"""
Intentionally weird and unconventional waypoint generation algorithms.

These algorithms explore unconventional approaches that might seem counterintuitive
but could potentially discover novel solutions. The tournament will filter out
the truly bad ones, but some might be surprisingly effective.
"""

import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional
import time

try:
    from .waypoint_generation import WaypointGenerator
    from .improved_waypoint_scoring import improved_score_waypoint_list
except ImportError:
    from waypoint_generation import WaypointGenerator
    from improved_waypoint_scoring import improved_score_waypoint_list


class ChaosWaypointGenerator(WaypointGenerator):
    """Completely random waypoint placement - pure chaos theory approach."""
    
    def __init__(self):
        super().__init__("Chaos")
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate completely random waypoints - sometimes chaos works."""
        height, width = screenshot.shape
        
        # Random number of waypoints (0 to 5)
        num_waypoints = random.randint(0, 5)
        waypoints = []
        
        for _ in range(num_waypoints):
            # Completely random position and size
            x = random.uniform(0, width)
            y = random.uniform(0, height) 
            radius = random.uniform(1.0, min(width, height) * 0.3)
            
            waypoints.append({
                'x': x,
                'y': y,
                'radius': radius
            })
        
        return waypoints


class AntiWaypointGenerator(WaypointGenerator):
    """Places waypoints in the WORST possible locations intentionally."""
    
    def __init__(self):
        super().__init__("Anti")
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Place waypoints in walls and other terrible locations."""
        height, width = screenshot.shape
        waypoints = []
        
        # Find walls and place waypoints there (terrible idea)
        wall_positions = [(x, y) for y, x in zip(*np.where(screenshot == 1))]
        
        if wall_positions:
            # Pick 1-3 wall positions
            num_waypoints = min(3, len(wall_positions))
            selected_walls = random.sample(wall_positions, num_waypoints)
            
            for x, y in selected_walls:
                waypoints.append({
                    'x': float(x),
                    'y': float(y),
                    'radius': random.uniform(2.0, 15.0)
                })
        
        return waypoints


class MegaWaypointGenerator(WaypointGenerator):
    """Creates one massive waypoint that covers most of the level."""
    
    def __init__(self):
        super().__init__("Mega")
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """One giant waypoint to rule them all."""
        height, width = screenshot.shape
        
        # Find center of passable area
        passable_y, passable_x = np.where(screenshot != 1)
        if len(passable_x) == 0:
            return []
        
        center_x = float(np.mean(passable_x))
        center_y = float(np.mean(passable_y))
        
        # Make it HUGE
        radius = min(width, height) * 0.4
        
        return [{
            'x': center_x,
            'y': center_y,
            'radius': radius
        }]


class FibonacciWaypointGenerator(WaypointGenerator):
    """Places waypoints according to Fibonacci spiral - nature-inspired."""
    
    def __init__(self):
        super().__init__("Fibonacci")
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Use golden ratio and Fibonacci spiral for waypoint placement."""
        height, width = screenshot.shape
        
        # Find center of level
        sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
        sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
        
        if not sources or not sinks:
            return []
        
        center_x = (np.mean([x for x, y in sources + sinks]))
        center_y = (np.mean([y for x, y in sources + sinks]))
        
        waypoints = []
        golden_ratio = (1 + math.sqrt(5)) / 2
        
        # Create Fibonacci spiral
        for i in range(1, 4):  # Max 3 waypoints
            angle = i * 2 * math.pi / golden_ratio
            distance = i * 15  # Fibonacci-ish spacing
            
            x = center_x + distance * math.cos(angle)
            y = center_y + distance * math.sin(angle)
            
            # Keep within bounds
            x = max(5, min(width - 5, x))
            y = max(5, min(height - 5, y))
            
            # Skip if it's a wall
            if screenshot[int(y), int(x)] == 1:
                continue
            
            waypoints.append({
                'x': x,
                'y': y,
                'radius': 8.0 + i * 2  # Growing radius
            })
        
        return waypoints


class MirrorWaypointGenerator(WaypointGenerator):
    """Creates symmetric waypoint patterns."""
    
    def __init__(self):
        super().__init__("Mirror")
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Create perfectly symmetric waypoint arrangements."""
        height, width = screenshot.shape
        
        center_x = width / 2
        center_y = height / 2
        
        waypoints = []
        
        # Create symmetric pairs
        for i in range(2):  # Max 2 pairs = 4 waypoints
            # Random offset from center
            offset_x = random.uniform(20, min(50, width/4))
            offset_y = random.uniform(20, min(50, height/4))
            
            # Create 4 symmetric points
            positions = [
                (center_x + offset_x, center_y + offset_y),
                (center_x - offset_x, center_y + offset_y),
                (center_x + offset_x, center_y - offset_y),
                (center_x - offset_x, center_y - offset_y),
            ]
            
            radius = random.uniform(5.0, 12.0)
            
            for x, y in positions:
                # Check bounds and walls
                if (5 <= x < width - 5 and 5 <= y < height - 5 and 
                    screenshot[int(y), int(x)] != 1):
                    waypoints.append({
                        'x': x,
                        'y': y,
                        'radius': radius
                    })
                    
                    # Only add one pair for now
                    if len(waypoints) >= 2:
                        return waypoints
        
        return waypoints


class PrimeNumberWaypointGenerator(WaypointGenerator):
    """Places waypoints at coordinates based on prime numbers."""
    
    def __init__(self):
        super().__init__("Prime")
        self.primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Use prime number sequences for waypoint coordinates."""
        height, width = screenshot.shape
        
        waypoints = []
        
        # Use prime numbers as coordinates (scaled to image)
        for i in range(min(3, len(self.primes) - 1)):
            x_prime = self.primes[i]
            y_prime = self.primes[i + 1]
            
            # Scale to image dimensions
            x = (x_prime / max(self.primes)) * (width - 10) + 5
            y = (y_prime / max(self.primes)) * (height - 10) + 5
            
            # Check if position is valid
            if screenshot[int(y), int(x)] != 1:
                waypoints.append({
                    'x': x,
                    'y': y,
                    'radius': float(self.primes[i % 5] / 2)  # Prime-based radius
                })
        
        return waypoints


class TimeBasedWaypointGenerator(WaypointGenerator):
    """Uses current time as seed for waypoint generation."""
    
    def __init__(self):
        super().__init__("TimeBased")
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints based on current system time."""
        height, width = screenshot.shape
        
        # Use milliseconds as pseudo-random seed
        time_seed = int(time.time() * 1000) % 10000
        random.seed(time_seed)
        
        # Extract digits for coordinates
        digits = [int(d) for d in str(time_seed)]
        
        waypoints = []
        
        for i in range(0, len(digits) - 1, 2):
            if len(waypoints) >= 3:
                break
                
            x_digit = digits[i]
            y_digit = digits[i + 1] if i + 1 < len(digits) else digits[0]
            
            # Scale digits to coordinates
            x = (x_digit / 9.0) * (width - 10) + 5
            y = (y_digit / 9.0) * (height - 10) + 5
            
            if screenshot[int(y), int(x)] != 1:
                waypoints.append({
                    'x': x,
                    'y': y,
                    'radius': 5.0 + (time_seed % 10)
                })
        
        return waypoints


class CornerMagnifierGenerator(WaypointGenerator):
    """Places tiny waypoints in every corner it can find."""
    
    def __init__(self):
        super().__init__("CornerMagnifier")
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Find all corners and place tiny waypoints there."""
        height, width = screenshot.shape
        waypoints = []
        
        # Scan for corner patterns
        for y in range(2, height - 2):
            for x in range(2, width - 2):
                if screenshot[y, x] != 1:  # Not a wall
                    # Check for corner patterns
                    neighbors = [
                        screenshot[y-1, x], screenshot[y+1, x],  # up, down
                        screenshot[y, x-1], screenshot[y, x+1],  # left, right
                        screenshot[y-1, x-1], screenshot[y-1, x+1],  # diagonals
                        screenshot[y+1, x-1], screenshot[y+1, x+1]
                    ]
                    
                    wall_count = sum(1 for n in neighbors if n == 1)
                    
                    # If surrounded by many walls, it's probably a corner
                    if wall_count >= 5:
                        waypoints.append({
                            'x': float(x),
                            'y': float(y),
                            'radius': 2.0  # Very tiny
                        })
                        
                        # Limit to prevent too many waypoints
                        if len(waypoints) >= 8:
                            return waypoints
        
        return waypoints


class EdgeHuggerGenerator(WaypointGenerator):
    """Places waypoints right along walls - hugging the edges."""
    
    def __init__(self):
        super().__init__("EdgeHugger")
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Place waypoints as close to walls as possible."""
        height, width = screenshot.shape
        waypoints = []
        
        # Find wall edges
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if screenshot[y, x] != 1:  # Not a wall
                    # Check if adjacent to wall
                    adjacent_to_wall = any([
                        screenshot[y-1, x] == 1,  # wall above
                        screenshot[y+1, x] == 1,  # wall below
                        screenshot[y, x-1] == 1,  # wall left
                        screenshot[y, x+1] == 1   # wall right
                    ])
                    
                    if adjacent_to_wall and random.random() < 0.03:  # Sample some edge positions
                        waypoints.append({
                            'x': float(x),
                            'y': float(y),
                            'radius': 3.0  # Small radius to hug edge
                        })
                        
                        if len(waypoints) >= 4:
                            return waypoints
        
        return waypoints


def create_weird_tournament():
    """Create a tournament with all the weird algorithms."""
    try:
        from .waypoint_generation import WaypointTournament, NullGenerator, CornerTurningGenerator
    except ImportError:
        from waypoint_generation import WaypointTournament, NullGenerator, CornerTurningGenerator
    
    tournament = WaypointTournament()
    
    # Add baseline algorithms
    tournament.add_generator(NullGenerator())
    tournament.add_generator(CornerTurningGenerator())
    
    # Add all weird algorithms
    tournament.add_generator(ChaosWaypointGenerator())
    tournament.add_generator(AntiWaypointGenerator())
    tournament.add_generator(MegaWaypointGenerator())
    tournament.add_generator(FibonacciWaypointGenerator())
    tournament.add_generator(MirrorWaypointGenerator())
    tournament.add_generator(PrimeNumberWaypointGenerator())
    tournament.add_generator(TimeBasedWaypointGenerator())
    tournament.add_generator(CornerMagnifierGenerator())
    tournament.add_generator(EdgeHuggerGenerator())
    
    return tournament


if __name__ == "__main__":
    # Test weird algorithms
    print("Testing weird waypoint algorithms...")
    
    from waypoint_test_runner import create_synthetic_test_cases
    
    test_cases = create_synthetic_test_cases()
    tournament = create_weird_tournament()
    
    print(f"Created weird tournament with {len(tournament.generators)} algorithms")
    
    # Test on L-shape case
    test_name, screenshot = test_cases[1]
    print(f"\nTesting weird algorithms on {test_name}:")
    
    for generator in tournament.generators:
        try:
            start_time = time.time()
            waypoints = generator.generate_waypoints(screenshot)
            duration = time.time() - start_time
            
            score = improved_score_waypoint_list(screenshot, waypoints, 
                                               penalize_skippable=True)
            
            print(f"  {generator.name:15} | {len(waypoints):2d} waypoints | {duration:5.3f}s | Score: {score:8.2f}")
            
        except Exception as e:
            print(f"  {generator.name:15} | ERROR: {e}")
    
    print("\nWeird algorithms test completed! Some might be surprisingly effective...")