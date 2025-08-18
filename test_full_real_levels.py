#!/usr/bin/env python3
"""
Full tournament test on real levels using ftlib.
"""

import sys
import os

# Add ftlib test directory to path
ftlib_test_path = os.path.join(os.path.dirname(__file__), 'ftlib', 'test')
sys.path.append(ftlib_test_path)

# Add py_autotweaker to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'py_autotweaker'))

from waypoint_test_runner import run_test_on_level_ids, load_maze_like_levels

def main():
    print("="*80)
    print("FULL REAL LEVEL TOURNAMENT")
    print("="*80)
    
    # Load level IDs
    level_ids = load_maze_like_levels('maze_like_levels.tsv')
    if not level_ids:
        print("✗ Could not load level IDs")
        return
    
    print(f"Loaded {len(level_ids)} level IDs from maze_like_levels.tsv")
    
    # Test on first 5 levels for now (to avoid long runtime)
    test_count = 5
    print(f"Testing on first {test_count} levels...")
    
    try:
        results = run_test_on_level_ids(level_ids[:test_count], 
                                      max_levels=test_count, 
                                      verbose=True)
        
        if "tournament_results" in results:
            print(f"\n🎉 Tournament completed successfully!")
            print(f"Tested {results['test_case_count']} valid levels")
            
            if results['failed_levels']:
                print(f"Failed to load {len(results['failed_levels'])} levels")
                
            # Show top performer
            tournament_results = results["tournament_results"]
            if tournament_results:
                best_name = min(tournament_results.keys(), 
                              key=lambda k: tournament_results[k]['total_score'])
                best_score = tournament_results[best_name]['total_score']
                print(f"\n🏆 Best performer: {best_name} (score: {best_score:.2f})")
        else:
            print(f"✗ Tournament failed: {results}")
            
    except Exception as e:
        print(f"✗ Tournament error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()