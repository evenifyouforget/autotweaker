#!/usr/bin/env python3
"""
Quick test for advanced waypoint generation algorithms.
"""

import sys
import os

# Add py_autotweaker to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'py_autotweaker'))

from waypoint_test_runner import create_synthetic_test_cases
from advanced_waypoint_generators import MedialAxisGenerator, VoronoiGenerator

def quick_test():
    print("="*60)
    print("QUICK ADVANCED ALGORITHM TEST")
    print("="*60)
    
    # Create a simple test case
    import numpy as np
    
    # Simple L-shaped corridor
    screenshot = np.ones((40, 60), dtype=np.uint8)  # All walls
    screenshot[10:15, 10:50] = 0  # Horizontal corridor
    screenshot[10:30, 45:50] = 0  # Vertical corridor
    screenshot[12, 12] = 3  # Source
    screenshot[27, 47] = 4  # Sink
    
    print("Testing on simple L-shaped corridor...")
    
    # Test each advanced algorithm
    algorithms = [
        MedialAxisGenerator(min_waypoint_distance=8.0, max_waypoints=5),
        VoronoiGenerator(min_distance_from_wall=3.0, max_waypoints=5)
    ]
    
    for algo in algorithms:
        try:
            print(f"\nTesting {algo.name}:")
            waypoints = algo.generate_waypoints(screenshot)
            
            from waypoint_scoring import score_waypoint_list, check_waypoint_non_skippability
            score = score_waypoint_list(screenshot, waypoints, penalize_skippable=True)
            is_valid = check_waypoint_non_skippability(screenshot, waypoints)
            
            print(f"  Generated {len(waypoints)} waypoints")
            print(f"  Score: {score:.2f}")
            print(f"  Valid (non-skippable): {'✓' if is_valid else '✗'}")
            
            if waypoints:
                print("  Waypoints:")
                for i, wp in enumerate(waypoints):
                    print(f"    {i+1}: ({wp['x']:.1f}, {wp['y']:.1f}) radius={wp['radius']:.1f}")
                    
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("QUICK TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    quick_test()