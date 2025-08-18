"""
Full tournament runner that uses the complete maze_like_levels.tsv database.

This module implements comprehensive testing on all 100 levels with proper
statistical analysis and results tracking.
"""

import sys
import os
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import statistics

# Add ftlib test directory to path
ftlib_test_path = os.path.join(os.path.dirname(__file__), '..', 'ftlib', 'test')
sys.path.append(ftlib_test_path)

try:
    # Try relative imports first (when run as module)
    from .waypoint_test_runner import (load_maze_like_levels, create_test_screenshot, 
                                      validate_screenshot_for_waypoints)
    from .waypoint_generation import create_default_tournament, WaypointTournament
    from .creative_waypoint_generators import create_creative_tournament
    from .improved_waypoint_scoring import improved_score_waypoint_list
except ImportError:
    # Fall back to absolute imports (when run as script)
    from waypoint_test_runner import (load_maze_like_levels, create_test_screenshot, 
                                     validate_screenshot_for_waypoints)
    from waypoint_generation import create_default_tournament, WaypointTournament
    from creative_waypoint_generators import create_creative_tournament
    from improved_waypoint_scoring import improved_score_waypoint_list


def load_all_test_levels(max_levels: Optional[int] = None, 
                        image_dimensions: Tuple[int, int] = (200, 145),
                        verbose: bool = True) -> Tuple[List[Tuple[str, np.ndarray]], List[Tuple[str, str]]]:
    """
    Load all test levels from maze_like_levels.tsv.
    
    Returns:
        Tuple of (successful_levels, failed_levels)
    """
    try:
        from get_design import retrieveLevel, designDomToStruct
        import numpy as np
    except ImportError:
        raise ImportError("ftlib get_design module required for full database testing")
    
    # Load level IDs
    tsv_path = os.path.join(os.path.dirname(__file__), '..', 'maze_like_levels.tsv')
    level_ids = load_maze_like_levels(tsv_path)
    
    if not level_ids:
        raise FileNotFoundError(f"Could not load level IDs from {tsv_path}")
    
    if max_levels:
        level_ids = level_ids[:max_levels]
    
    if verbose:
        print(f"Loading {len(level_ids)} levels from maze_like_levels.tsv...")
    
    successful_levels = []
    failed_levels = []
    
    # Load and validate levels
    for i, level_id in enumerate(level_ids):
        try:
            if verbose and (i + 1) % 20 == 0:
                print(f"  Progress: {i + 1}/{len(level_ids)} levels...")
            
            # Get design data
            level_dom = retrieveLevel(level_id, is_design=False)
            if level_dom is None:
                failed_levels.append((level_id, "Could not retrieve level"))
                continue
            
            design_struct = designDomToStruct(level_dom)
            if design_struct is None:
                failed_levels.append((level_id, "Could not parse level"))
                continue
            
            # Create screenshot
            screenshot = create_test_screenshot(design_struct, image_dimensions)
            
            # Validate screenshot
            is_valid, reason = validate_screenshot_for_waypoints(screenshot)
            if not is_valid:
                failed_levels.append((level_id, reason))
                continue
            
            successful_levels.append((level_id, screenshot))
            
        except Exception as e:
            failed_levels.append((level_id, f"Error: {str(e)}"))
            continue
    
    if verbose:
        print(f"Successfully loaded {len(successful_levels)} levels ({len(failed_levels)} failed)")
        if failed_levels and verbose:
            print("Sample failed levels:")
            for level_id, reason in failed_levels[:3]:
                print(f"  {level_id}: {reason}")
            if len(failed_levels) > 3:
                print(f"  ... and {len(failed_levels) - 3} more")
    
    return successful_levels, failed_levels


def run_comprehensive_tournament(max_levels: Optional[int] = None, 
                               use_creative: bool = False,
                               use_improved_scoring: bool = True,
                               verbose: bool = True,
                               save_results: bool = True) -> Dict[str, Any]:
    """
    Run comprehensive tournament on the full level database.
    
    Args:
        max_levels: Maximum levels to test (None for all 100)
        use_creative: Include creative algorithms (slower)
        use_improved_scoring: Use improved scoring function
        verbose: Print detailed progress
        save_results: Save results to JSON file
    
    Returns:
        Comprehensive results dictionary
    """
    start_time = time.time()
    
    # Load test levels
    print("="*80)
    print("COMPREHENSIVE WAYPOINT TOURNAMENT")
    print("="*80)
    
    test_levels, failed_levels = load_all_test_levels(max_levels=max_levels, verbose=verbose)
    
    if not test_levels:
        return {"error": "No valid test levels loaded"}
    
    # Create tournament
    if use_creative:
        try:
            tournament = create_creative_tournament()
            print(f"Using creative tournament with {len(tournament.generators)} algorithms")
        except ImportError:
            print("Warning: Creative tournament not available, using default")
            tournament = create_default_tournament()
            print(f"Using default tournament with {len(tournament.generators)} algorithms")
    else:
        tournament = create_default_tournament()
        print(f"Using default tournament with {len(tournament.generators)} algorithms")
    
    # Override scoring function if requested
    if use_improved_scoring:
        print("Using improved scoring function")
        original_score_func = None
        # We'll use improved scoring in the generators directly
    
    # Run tournament
    print(f"\nRunning tournament on {len(test_levels)} levels...")
    
    results = tournament.run_tournament(test_levels, verbose=verbose)
    
    if verbose:
        tournament.print_final_rankings()
    
    # Calculate additional statistics
    comprehensive_results = {
        "tournament_results": results,
        "test_configuration": {
            "total_levels_attempted": len(test_levels) + len(failed_levels),
            "successful_levels": len(test_levels),
            "failed_levels": len(failed_levels),
            "max_levels_requested": max_levels,
            "use_creative": use_creative,
            "use_improved_scoring": use_improved_scoring,
            "image_dimensions": (200, 145),
            "execution_time_seconds": time.time() - start_time
        },
        "failed_level_analysis": _analyze_failed_levels(failed_levels),
        "performance_analysis": _analyze_algorithm_performance(results),
        "level_difficulty_analysis": _analyze_level_difficulty(test_levels, tournament, use_improved_scoring)
    }
    
    # Save results
    if save_results:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"tournament_results_{timestamp}.json"
        filepath = os.path.join(os.path.dirname(__file__), '..', 'results', filename)
        
        # Create results directory
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save to file
        try:
            with open(filepath, 'w') as f:
                json.dump(comprehensive_results, f, indent=2, default=str)
            print(f"\nResults saved to: {filepath}")
        except Exception as e:
            print(f"Warning: Could not save results to file: {e}")
    
    return comprehensive_results


def _analyze_failed_levels(failed_levels: List[Tuple[str, str]]) -> Dict[str, Any]:
    """Analyze why levels failed to load."""
    if not failed_levels:
        return {"total_failed": 0}
    
    # Categorize failure reasons
    failure_categories = defaultdict(int)
    for level_id, reason in failed_levels:
        if "not reach" in reason.lower():
            failure_categories["connectivity_issues"] += 1
        elif "source" in reason.lower() or "sink" in reason.lower():
            failure_categories["missing_elements"] += 1
        elif "error" in reason.lower():
            failure_categories["loading_errors"] += 1
        else:
            failure_categories["other"] += 1
    
    return {
        "total_failed": len(failed_levels),
        "failure_categories": dict(failure_categories),
        "sample_failures": failed_levels[:5]  # First 5 for reference
    }


def _analyze_algorithm_performance(results: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze algorithm performance patterns."""
    if not results:
        return {}
    
    analysis = {
        "algorithm_rankings": [],
        "score_statistics": {},
        "performance_categories": {
            "excellent": [],      # Score < 10
            "good": [],           # Score 10-100  
            "mediocre": [],       # Score 100-1000
            "poor": []            # Score > 1000
        },
        "reliability_analysis": {}
    }
    
    # Rank algorithms by average score
    for name, data in results.items():
        avg_score = data.get('avg_score', float('inf'))
        total_score = data.get('total_score', float('inf'))
        error_rate = data['error_count'] / (len(data['scores']) + data['error_count']) if data['scores'] or data['error_count'] else 0
        
        analysis["algorithm_rankings"].append({
            "name": name,
            "avg_score": avg_score,
            "total_score": total_score,
            "error_rate": error_rate,
            "skippable_rate": data['skippable_count'] / max(1, len(data['scores'])),
            "avg_waypoints": data.get('avg_waypoints', 0),
            "avg_time": data.get('avg_time', 0)
        })
        
        # Categorize performance
        if avg_score < 10:
            analysis["performance_categories"]["excellent"].append(name)
        elif avg_score < 100:
            analysis["performance_categories"]["good"].append(name)
        elif avg_score < 1000:
            analysis["performance_categories"]["mediocre"].append(name)
        else:
            analysis["performance_categories"]["poor"].append(name)
        
        # Score statistics
        if data['scores']:
            analysis["score_statistics"][name] = {
                "mean": statistics.mean(data['scores']),
                "median": statistics.median(data['scores']),
                "std_dev": statistics.stdev(data['scores']) if len(data['scores']) > 1 else 0,
                "min": min(data['scores']),
                "max": max(data['scores'])
            }
    
    # Sort rankings by average score
    analysis["algorithm_rankings"].sort(key=lambda x: x["avg_score"])
    
    return analysis


def _analyze_level_difficulty(test_levels: List[Tuple[str, np.ndarray]], 
                            tournament: WaypointTournament,
                            use_improved_scoring: bool) -> Dict[str, Any]:
    """Analyze which levels are most difficult for waypoint generation."""
    import numpy as np
    
    level_difficulty = []
    
    # Sample a few levels for difficulty analysis
    sample_levels = test_levels[:min(10, len(test_levels))]
    
    for level_id, screenshot in sample_levels:
        level_scores = []
        
        # Test each algorithm on this level
        for generator in tournament.generators[:3]:  # Test first 3 for speed
            try:
                waypoints = generator.generate_waypoints(screenshot)
                if use_improved_scoring:
                    score = improved_score_waypoint_list(screenshot, waypoints, penalize_skippable=True)
                else:
                    from waypoint_scoring import score_waypoint_list
                    score = score_waypoint_list(screenshot, waypoints, penalize_skippable=True)
                level_scores.append(score)
            except:
                level_scores.append(1000.0)  # High penalty for errors
        
        if level_scores:
            difficulty_metrics = {
                "level_id": level_id,
                "avg_score": statistics.mean(level_scores),
                "min_score": min(level_scores),
                "max_score": max(level_scores),
                "score_variance": statistics.variance(level_scores) if len(level_scores) > 1 else 0
            }
            level_difficulty.append(difficulty_metrics)
    
    # Sort by average difficulty (higher scores = more difficult)
    level_difficulty.sort(key=lambda x: x["avg_score"], reverse=True)
    
    return {
        "sample_size": len(sample_levels),
        "most_difficult_levels": level_difficulty[:5],
        "easiest_levels": level_difficulty[-5:],
        "difficulty_distribution": {
            "very_easy": len([l for l in level_difficulty if l["avg_score"] < 10]),
            "easy": len([l for l in level_difficulty if 10 <= l["avg_score"] < 50]),
            "medium": len([l for l in level_difficulty if 50 <= l["avg_score"] < 200]),
            "hard": len([l for l in level_difficulty if 200 <= l["avg_score"] < 1000]),
            "very_hard": len([l for l in level_difficulty if l["avg_score"] >= 1000])
        }
    }


def main():
    """Main entry point for comprehensive tournament."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive waypoint tournament on full database')
    parser.add_argument('--max-levels', type=int, help='Maximum levels to test (default: all 100)')
    parser.add_argument('--creative', action='store_true', help='Include creative algorithms (slower)')
    parser.add_argument('--fast', action='store_true', help='Fast mode: limit levels and algorithms')
    parser.add_argument('--quiet', action='store_true', help='Reduce output verbosity')
    
    args = parser.parse_args()
    
    # Configure based on arguments
    max_levels = args.max_levels
    if args.fast:
        max_levels = 20  # Fast mode: only 20 levels
        use_creative = True  # Still use creative algorithms in fast mode
    else:
        use_creative = args.creative
    
    verbose = not args.quiet
    
    try:
        results = run_comprehensive_tournament(
            max_levels=max_levels,
            use_creative=use_creative,
            use_improved_scoring=True,
            verbose=verbose,
            save_results=True
        )
        
        if "error" in results:
            print(f"Tournament failed: {results['error']}")
            return 1
        
        # Print summary
        print(f"\n🎉 Comprehensive tournament completed!")
        config = results["test_configuration"]
        print(f"Tested {config['successful_levels']} levels in {config['execution_time_seconds']:.1f} seconds")
        
        # Show top performers
        perf_analysis = results.get("performance_analysis", {})
        if perf_analysis and "algorithm_rankings" in perf_analysis:
            rankings = perf_analysis["algorithm_rankings"]
            print(f"\n🏆 Top 3 algorithms:")
            for i, algo in enumerate(rankings[:3], 1):
                print(f"  {i}. {algo['name']} (avg score: {algo['avg_score']:.2f})")
        
        return 0
        
    except Exception as e:
        print(f"Tournament failed: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())