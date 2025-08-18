#!/usr/bin/env python3
"""
Tournament runner entry point for waypoint generation algorithms.

This module provides command-line interface for running waypoint generation
tournaments on synthetic or real levels.
"""

import sys
import os
import argparse
from typing import Optional, List

# Add ftlib test directory to path
ftlib_test_path = os.path.join(os.path.dirname(__file__), '..', 'ftlib', 'test')
sys.path.append(ftlib_test_path)

try:
    # Try relative imports first (when run as module)
    from .waypoint_test_runner import (create_synthetic_test_cases, run_test_on_level_ids, 
                                      load_maze_like_levels, test_synthetic_levels)
    from .waypoint_generation import create_default_tournament
    from .advanced_waypoint_generators import create_advanced_tournament
except ImportError:
    # Fall back to absolute imports (when run as script)
    from waypoint_test_runner import (create_synthetic_test_cases, run_test_on_level_ids, 
                                     load_maze_like_levels, test_synthetic_levels)
    from waypoint_generation import create_default_tournament
    from advanced_waypoint_generators import create_advanced_tournament


def run_synthetic_tournament(use_advanced: bool = False, verbose: bool = True) -> dict:
    """Run tournament on synthetic test cases."""
    print("Running tournament on synthetic levels...")
    
    if use_advanced:
        try:
            from quick_creative_generators import create_quick_creative_tournament
            tournament = create_quick_creative_tournament()
            print(f"Using quick creative tournament with {len(tournament.generators)} algorithms")
        except ImportError:
            try:
                from creative_waypoint_generators import create_creative_tournament
                tournament = create_creative_tournament()
                print(f"Using full creative tournament with {len(tournament.generators)} algorithms")
            except ImportError:
                print("Warning: Creative algorithms not available, using advanced tournament")
                tournament = create_advanced_tournament()
    else:
        tournament = create_default_tournament()
    
    test_cases = create_synthetic_test_cases()
    results = tournament.run_tournament(test_cases, verbose=verbose)
    
    if verbose:
        tournament.print_final_rankings()
    
    return results


def run_real_level_tournament(max_levels: int = 10, use_advanced: bool = False, 
                            verbose: bool = True) -> dict:
    """Run tournament on real levels from maze_like_levels.tsv."""
    print(f"Running tournament on up to {max_levels} real levels...")
    
    if use_advanced:
        print("Using full comprehensive tournament system with creative algorithms...")
        try:
            from full_tournament_runner import run_comprehensive_tournament
            return run_comprehensive_tournament(
                max_levels=max_levels, 
                use_creative=True,
                use_improved_scoring=True,
                verbose=verbose,
                save_results=True
            )
        except ImportError:
            print("Warning: Comprehensive tournament not available, falling back to basic")
    
    # Fallback to basic tournament
    tsv_path = os.path.join(os.path.dirname(__file__), '..', 'maze_like_levels.tsv')
    level_ids = load_maze_like_levels(tsv_path)
    
    if not level_ids:
        raise FileNotFoundError(f"Could not load level IDs from {tsv_path}")
    
    print(f"Found {len(level_ids)} level IDs in maze_like_levels.tsv")
    
    # Run tournament
    results = run_test_on_level_ids(level_ids, max_levels=max_levels, verbose=verbose)
    
    return results


def run_mixed_tournament(max_real_levels: int = 5, use_advanced: bool = False,
                        verbose: bool = True) -> dict:
    """Run tournament on both synthetic and real levels."""
    print("Running mixed tournament (synthetic + real levels)...")
    
    if use_advanced:
        tournament = create_advanced_tournament()
    else:
        tournament = create_default_tournament()
    
    # Combine synthetic and real test cases
    synthetic_cases = create_synthetic_test_cases()
    
    # Load some real levels
    tsv_path = os.path.join(os.path.dirname(__file__), '..', 'maze_like_levels.tsv')
    level_ids = load_maze_like_levels(tsv_path)
    
    real_cases = []
    if level_ids:
        try:
            from waypoint_test_runner import create_test_screenshot
            from get_design import FCDesignStruct, retrieveLevel, designDomToStruct
            
            print(f"Loading {min(max_real_levels, len(level_ids))} real levels...")
            
            for i, level_id in enumerate(level_ids[:max_real_levels]):
                try:
                    level_dom = retrieveLevel(level_id, is_design=False)
                    if level_dom is None:
                        continue
                    design = designDomToStruct(level_dom)
                    if design is None:
                        continue
                    
                    screenshot = create_test_screenshot(design, (200, 145))
                    
                    # Validate screenshot
                    from waypoint_test_runner import validate_screenshot_for_waypoints
                    is_valid, reason = validate_screenshot_for_waypoints(screenshot)
                    if is_valid:
                        real_cases.append((f"real_{level_id}", screenshot))
                        if verbose:
                            print(f"  Loaded level {level_id} ({i+1}/{max_real_levels})")
                    elif verbose:
                        print(f"  Skipped level {level_id}: {reason}")
                        
                except Exception as e:
                    if verbose:
                        print(f"  Failed to load level {level_id}: {e}")
                    continue
        
        except ImportError:
            print("Warning: Could not load real levels (ftlib not available)")
    
    # Combine test cases
    all_cases = synthetic_cases + real_cases
    print(f"Running tournament on {len(all_cases)} total cases ({len(synthetic_cases)} synthetic + {len(real_cases)} real)")
    
    # Run tournament
    results = tournament.run_tournament(all_cases, verbose=verbose)
    
    if verbose:
        tournament.print_final_rankings()
    
    return results


def main():
    """Main entry point for command-line usage."""
    parser = argparse.ArgumentParser(description='Run waypoint generation tournament')
    
    parser.add_argument('--mode', choices=['synthetic', 'real', 'mixed', 'list'], default='synthetic',
                       help='Tournament mode (default: synthetic)')
    parser.add_argument('--max-levels', type=int, default=10,
                       help='Maximum number of real levels to test (default: 10, use 100 for full database)')
    parser.add_argument('--advanced', action='store_true',
                       help='Include all creative algorithms and comprehensive analysis (slower)')
    parser.add_argument('--full', action='store_true',
                       help='Test all 100 levels with comprehensive analysis (equivalent to --max-levels 100 --advanced)')
    parser.add_argument('--comprehensive', action='store_true',
                       help='Use full comprehensive algorithms (slower, more thorough)')
    parser.add_argument('--quiet', action='store_true',
                       help='Reduce output verbosity')
    parser.add_argument('--list-algorithms', action='store_true',
                       help='List available algorithms and exit')
    
    args = parser.parse_args()
    
    # Handle --full flag
    if args.full:
        args.max_levels = 100
        args.advanced = True
        print("Full tournament mode: Testing all 100 levels with creative algorithms")
    
    if args.list_algorithms or args.mode == 'list':
        print("Available Algorithms:")
        print("Basic algorithms (always included):")
        print("  - Null: Empty waypoint list (baseline)")
        print("  - CornerTurning: Recursive corner detection")
        
        if args.advanced:
            print("Creative algorithms (--advanced flag):")
            print("  - Genetic: Evolutionary optimization of waypoint populations")
            print("  - FlowField: Potential field analysis with critical point detection")
            print("  - SwarmIntelligence: Particle swarm optimization")  
            print("  - AdaptiveRandom: Random sampling with learning")
            print("  - ImprovedCornerTurning: Physics-based balloon expansion")
            print("  - MedialAxis: Skeleton-based waypoint placement")
            print("  - Voronoi: Distance transform optimization")
            print("  - OptimizedSearch: Simulated annealing")
        else:
            print("\nUse --advanced to see all creative algorithms (8+ total)")
        return
    
    verbose = not args.quiet
    
    try:
        if args.mode == 'list':
            return  # Already handled above
        elif args.mode == 'synthetic':
            results = run_synthetic_tournament(use_advanced=args.advanced, verbose=verbose)
        elif args.mode == 'real':
            results = run_real_level_tournament(max_levels=args.max_levels, 
                                              use_advanced=args.advanced, verbose=verbose)
        elif args.mode == 'mixed':
            results = run_mixed_tournament(max_real_levels=args.max_levels,
                                         use_advanced=args.advanced, verbose=verbose)
        
        if verbose:
            print(f"\n🎉 Tournament completed successfully!")
            if "tournament_results" in results:
                total_cases = results.get('test_case_count', 0)
                print(f"Tested {total_cases} levels")
                
                # Show best performer
                tournament_results = results["tournament_results"]
                if tournament_results:
                    best_name = min(tournament_results.keys(), 
                                  key=lambda k: tournament_results[k]['total_score'])
                    best_score = tournament_results[best_name]['total_score']
                    print(f"🏆 Best performer: {best_name} (total score: {best_score:.2f})")
        
    except Exception as e:
        print(f"❌ Tournament failed: {e}", file=sys.stderr)
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()