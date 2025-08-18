#!/usr/bin/env python3
"""
Enhanced test script for advanced waypoint generation algorithms.

This script tests both basic and advanced waypoint generation algorithms
including medial axis, Voronoi, and optimization-based approaches.
"""

import sys
import os
import numpy as np

# Add py_autotweaker to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'py_autotweaker'))

from waypoint_test_runner import create_synthetic_test_cases, test_synthetic_levels
from advanced_waypoint_generators import create_advanced_tournament
import matplotlib.pyplot as plt

def create_more_complex_test_cases():
    """Create more challenging synthetic test cases."""
    test_cases = []
    
    # Test case 1: Complex maze
    screenshot1 = np.ones((80, 120), dtype=np.uint8)  # All walls
    # Create a complex maze pattern
    for y in range(10, 70, 10):
        screenshot1[y:y+5, 10:110] = 0  # Horizontal corridors
    for x in range(20, 100, 20):
        screenshot1[10:70, x:x+5] = 0   # Vertical corridors
    # Add some obstacles
    screenshot1[25:35, 40:60] = 1
    screenshot1[45:55, 70:90] = 1
    # Add source and sink
    screenshot1[15, 15] = 3  # source
    screenshot1[65, 105] = 4  # sink
    test_cases.append(("complex_maze", screenshot1))
    
    # Test case 2: Narrow passages with wide areas
    screenshot2 = np.zeros((60, 100), dtype=np.uint8)
    # Add borders
    screenshot2[0, :] = 1
    screenshot2[-1, :] = 1
    screenshot2[:, 0] = 1
    screenshot2[:, -1] = 1
    # Create narrow passages connecting wide areas
    screenshot2[20:40, 30:35] = 1  # Narrow passage
    screenshot2[20:40, 65:70] = 1  # Another narrow passage
    # Add source and sink in wide areas
    screenshot2[30, 15] = 3  # source in left area
    screenshot2[30, 85] = 4  # sink in right area
    test_cases.append(("narrow_passages", screenshot2))
    
    # Test case 3: Multiple path options
    screenshot3 = np.zeros((50, 90), dtype=np.uint8)
    # Add borders
    screenshot3[0, :] = 1
    screenshot3[-1, :] = 1
    screenshot3[:, 0] = 1
    screenshot3[:, -1] = 1
    # Create diamond-shaped obstacle in center
    for i in range(20):
        screenshot3[15+i, 35+i:55-i] = 1
        screenshot3[35-i, 35+i:55-i] = 1
    # Add source and sink
    screenshot3[25, 10] = 3  # source
    screenshot3[25, 80] = 4  # sink
    test_cases.append(("multiple_paths", screenshot3))
    
    return test_cases

def visualize_waypoints(screenshot, waypoints, title="Waypoint Visualization"):
    """Visualize a screenshot with waypoints overlaid."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create color map for screenshot
    color_map = np.zeros((*screenshot.shape, 3))
    color_map[screenshot == 0] = [0.9, 0.9, 0.9]  # Passable (light gray)
    color_map[screenshot == 1] = [0.2, 0.2, 0.2]  # Walls (dark gray)
    color_map[screenshot == 3] = [0.0, 1.0, 0.0]  # Source (green)
    color_map[screenshot == 4] = [1.0, 0.0, 0.0]  # Sink (red)
    
    ax.imshow(color_map, origin='upper')
    
    # Draw waypoints
    for i, wp in enumerate(waypoints):
        circle = plt.Circle((wp['x'], wp['y']), wp['radius'], 
                          fill=False, color='blue', linewidth=2)
        ax.add_patch(circle)
        ax.text(wp['x'], wp['y'], str(i+1), color='blue', 
               ha='center', va='center', fontweight='bold', fontsize=12)
    
    ax.set_title(title)
    ax.set_xlim(0, screenshot.shape[1])
    ax.set_ylim(screenshot.shape[0], 0)  # Flip y-axis
    return fig

def detailed_algorithm_comparison():
    """Run detailed comparison of algorithms with visualization."""
    print("="*80)
    print("DETAILED ALGORITHM COMPARISON")
    print("="*80)
    
    # Create test cases
    basic_cases = create_synthetic_test_cases()
    complex_cases = create_more_complex_test_cases()
    all_cases = basic_cases + complex_cases
    
    print(f"\nTesting {len(all_cases)} test cases...")
    
    # Run tournament
    tournament = create_advanced_tournament()
    results = tournament.run_tournament(all_cases, verbose=True)
    
    print("\n" + "="*80)
    print("DETAILED RESULTS ANALYSIS")
    print("="*80)
    
    # Analyze results per test case
    for test_name, screenshot in all_cases:
        print(f"\nTest Case: {test_name}")
        print("-" * 50)
        
        # Run each generator individually for this test case
        for generator in tournament.generators:
            try:
                waypoints = generator.generate_waypoints(screenshot)
                from waypoint_scoring import score_waypoint_list, check_waypoint_non_skippability
                
                score = score_waypoint_list(screenshot, waypoints, penalize_skippable=True)
                is_valid = check_waypoint_non_skippability(screenshot, waypoints)
                
                print(f"  {generator.name:20} | {len(waypoints):2d} waypoints | "
                      f"Score: {score:8.2f} | Valid: {'✓' if is_valid else '✗'}")
                
                # Optionally save visualization for best performers
                if score < 10 and len(waypoints) > 0 and generator.name != "Null":
                    try:
                        fig = visualize_waypoints(screenshot, waypoints, 
                                                f"{test_name} - {generator.name}")
                        plt.savefig(f"/tmp/{test_name}_{generator.name.lower()}_waypoints.png", 
                                  dpi=100, bbox_inches='tight')
                        plt.close(fig)
                    except Exception as e:
                        print(f"    (Visualization failed: {e})")
                        
            except Exception as e:
                print(f"  {generator.name:20} | ERROR: {e}")
    
    return results

def main():
    print("="*80)
    print("ADVANCED WAYPOINT GENERATION SYSTEM TEST")
    print("="*80)
    
    print("\n1. Basic synthetic test...")
    try:
        basic_results = test_synthetic_levels(verbose=False)
        print(f"✓ Basic test completed - {basic_results['test_case_count']} cases")
    except Exception as e:
        print(f"✗ Basic test failed: {e}")
    
    print("\n2. Advanced algorithm comparison...")
    try:
        detailed_results = detailed_algorithm_comparison()
        print("✓ Advanced comparison completed")
        
        # Print summary of best performers
        if "tournament_results" in detailed_results:
            results = detailed_results["tournament_results"]
            best_generator = min(results.items(), key=lambda x: x[1]['total_score'])
            print(f"\n🏆 Best performing algorithm: {best_generator[0]}")
            print(f"   Total score: {best_generator[1]['total_score']:.2f}")
            print(f"   Average waypoints: {best_generator[1].get('avg_waypoints', 0):.1f}")
            
    except Exception as e:
        print(f"✗ Advanced comparison failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*80)
    print("ADVANCED TEST COMPLETED")
    print("="*80)
    
    print("\nVisualization files (if generated) saved to /tmp/")
    print("Advanced algorithms tested:")
    print("  - MedialAxis: Uses skeleton/medial axis of passable areas")
    print("  - Voronoi: Places waypoints at maximal distance from walls")
    print("  - OptimizedSearch: Uses simulated annealing optimization")

if __name__ == "__main__":
    main()