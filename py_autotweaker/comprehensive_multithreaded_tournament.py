#!/usr/bin/env python3
"""
Comprehensive multithreaded tournament system.

Integrates all waypoint generation algorithms (basic, creative, weird) with
high-performance multithreaded execution, timeouts, and comprehensive analysis.
"""

import os
import sys
import time
import json
from typing import List, Dict, Tuple, Optional, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from .multithreaded_tournament import MultithreadedTournament, TournamentConfig
    from .waypoint_generation import create_default_tournament
    from .creative_waypoint_generators import create_creative_tournament
    from .quick_creative_generators import create_quick_creative_tournament
    from .weird_waypoint_generators import create_weird_tournament
except ImportError:
    from multithreaded_tournament import MultithreadedTournament, TournamentConfig
    from waypoint_generation import create_default_tournament
    from creative_waypoint_generators import create_creative_tournament
    from quick_creative_generators import create_quick_creative_tournament
    from weird_waypoint_generators import create_weird_tournament


def create_comprehensive_tournament(include_creative: bool = True,
                                  include_weird: bool = True,
                                  include_quick: bool = False,
                                  timeout_per_algorithm: float = 10.0,
                                  max_workers: Optional[int] = None) -> MultithreadedTournament:
    """
    Create a comprehensive tournament with all available algorithms.
    
    Args:
        include_creative: Include sophisticated creative algorithms
        include_weird: Include weird/experimental algorithms  
        include_quick: Include quick creative algorithms (faster but simpler)
        timeout_per_algorithm: Timeout in seconds for each algorithm run
        max_workers: Number of worker threads (None = auto-detect)
    
    Returns:
        Configured MultithreadedTournament
    """
    
    config = TournamentConfig(
        max_workers=max_workers,
        timeout_per_algorithm=timeout_per_algorithm,
        use_subprocess=True,  # Always use subprocess for safety
        verbose=True
    )
    
    tournament = MultithreadedTournament(config)
    
    # Always include basic algorithms
    basic_tournament = create_default_tournament()
    for generator in basic_tournament.generators:
        tournament.add_generator(generator)
    
    # Add creative algorithms
    if include_creative:
        try:
            if include_quick:
                creative_tournament = create_quick_creative_tournament()
                print(f"Added {len(creative_tournament.generators)} quick creative algorithms")
            else:
                creative_tournament = create_creative_tournament()
                print(f"Added {len(creative_tournament.generators)} full creative algorithms")
            
            for generator in creative_tournament.generators:
                # Avoid duplicates from basic tournament
                if not any(g.name == generator.name for g in tournament.generators):
                    tournament.add_generator(generator)
                    
        except ImportError as e:
            print(f"Warning: Could not load creative algorithms: {e}")
    
    # Add weird algorithms
    if include_weird:
        try:
            weird_tournament = create_weird_tournament()
            print(f"Added {len(weird_tournament.generators)} weird algorithms")
            
            for generator in weird_tournament.generators:
                # Avoid duplicates
                if not any(g.name == generator.name for g in tournament.generators):
                    tournament.add_generator(generator)
                    
        except ImportError as e:
            print(f"Warning: Could not load weird algorithms: {e}")
    
    print(f"Comprehensive tournament created with {len(tournament.generators)} total algorithms")
    return tournament


def run_comprehensive_tournament_analysis(test_cases: List[Tuple[str, Any]],
                                        include_creative: bool = True,
                                        include_weird: bool = True,
                                        timeout_per_algorithm: float = 10.0,
                                        save_results: bool = True,
                                        results_dir: str = "results") -> Dict[str, Any]:
    """
    Run a comprehensive tournament with full analysis and result saving.
    
    Args:
        test_cases: List of (test_case_name, screenshot) tuples
        include_creative: Include creative algorithms
        include_weird: Include weird algorithms
        timeout_per_algorithm: Timeout per algorithm in seconds
        save_results: Save detailed results to JSON
        results_dir: Directory to save results
        
    Returns:
        Comprehensive results dictionary
    """
    
    print("="*80)
    print("COMPREHENSIVE MULTITHREADED WAYPOINT TOURNAMENT")
    print("="*80)
    
    start_time = time.time()
    
    # Create tournament
    tournament = create_comprehensive_tournament(
        include_creative=include_creative,
        include_weird=include_weird,
        include_quick=False,  # Use full algorithms for comprehensive analysis
        timeout_per_algorithm=timeout_per_algorithm
    )
    
    print(f"\nTournament configuration:")
    print(f"  Test cases: {len(test_cases)}")
    print(f"  Algorithms: {len(tournament.generators)}")
    print(f"  Timeout per algorithm: {timeout_per_algorithm}s")
    print(f"  Max workers: {tournament.config.max_workers}")
    print(f"  Total algorithm runs: {len(test_cases) * len(tournament.generators)}")
    
    estimated_time = (len(test_cases) * len(tournament.generators) * timeout_per_algorithm) / tournament.config.max_workers
    print(f"  Estimated max time: {estimated_time/60:.1f} minutes")
    
    # Run tournament
    results = tournament.run_tournament(test_cases)
    
    # Print results
    tournament.print_results(results)
    
    # Enhanced analysis
    enhanced_results = enhance_tournament_analysis(results, test_cases)
    
    # Save results
    if save_results:
        save_tournament_results(enhanced_results, results_dir)
    
    total_time = time.time() - start_time
    print(f"\nComprehensive tournament completed in {total_time:.2f}s")
    
    return enhanced_results


def enhance_tournament_analysis(results: Dict[str, Any], 
                               test_cases: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """Add enhanced analysis to tournament results."""
    
    enhanced = results.copy()
    
    # Algorithm categorization
    basic_algorithms = ["Null", "CornerTurning"]
    creative_algorithms = ["Genetic", "FlowField", "SwarmIntelligence", "AdaptiveRandom", 
                          "ImprovedCornerTurning", "QuickGenetic", "QuickFlowField",
                          "QuickSwarm", "QuickAdaptive"]
    weird_algorithms = ["Chaos", "Anti", "Mega", "Fibonacci", "Mirror", "Prime", 
                       "TimeBased", "CornerMagnifier", "EdgeHugger"]
    
    # Categorize results
    tournament_results = results['tournament_results']
    categories = {
        'basic': {},
        'creative': {},
        'weird': {}
    }
    
    for alg_name, alg_data in tournament_results.items():
        if alg_name in basic_algorithms:
            categories['basic'][alg_name] = alg_data
        elif alg_name in creative_algorithms:
            categories['creative'][alg_name] = alg_data
        elif alg_name in weird_algorithms:
            categories['weird'][alg_name] = alg_data
    
    # Performance comparison by category
    category_analysis = {}
    for cat_name, cat_algorithms in categories.items():
        if not cat_algorithms:
            continue
            
        successful_algs = [name for name, data in cat_algorithms.items() 
                          if data['successful_runs'] > 0]
        
        if successful_algs:
            best_alg = min(successful_algs, key=lambda name: cat_algorithms[name]['avg_score'])
            avg_category_score = sum(cat_algorithms[name]['avg_score'] 
                                   for name in successful_algs) / len(successful_algs)
        else:
            best_alg = None
            avg_category_score = float('inf')
        
        category_analysis[cat_name] = {
            'algorithm_count': len(cat_algorithms),
            'successful_count': len(successful_algs),
            'best_algorithm': best_alg,
            'avg_category_score': avg_category_score,
            'success_rate': len(successful_algs) / len(cat_algorithms) if cat_algorithms else 0
        }
    
    enhanced['category_analysis'] = category_analysis
    enhanced['algorithm_categories'] = categories
    
    # Surprise findings (weird algorithms that outperform basic ones)
    surprise_findings = []
    if categories['basic'] and categories['weird']:
        basic_best_score = min(data['avg_score'] for data in categories['basic'].values() 
                              if data['successful_runs'] > 0) if any(data['successful_runs'] > 0 
                              for data in categories['basic'].values()) else float('inf')
        
        for alg_name, alg_data in categories['weird'].items():
            if (alg_data['successful_runs'] > 0 and 
                alg_data['avg_score'] < basic_best_score):
                surprise_findings.append({
                    'algorithm': alg_name,
                    'score': alg_data['avg_score'],
                    'improvement': basic_best_score - alg_data['avg_score']
                })
    
    enhanced['surprise_findings'] = surprise_findings
    
    return enhanced


def save_tournament_results(results: Dict[str, Any], results_dir: str):
    """Save tournament results to JSON file with timestamp."""
    
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"comprehensive_tournament_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nDetailed results saved to: {filepath}")
    except Exception as e:
        print(f"Warning: Could not save results: {e}")


def main():
    """Main entry point for comprehensive tournament."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive multithreaded waypoint tournament')
    parser.add_argument('--synthetic', action='store_true', help='Run on synthetic test cases')
    parser.add_argument('--real', action='store_true', help='Run on real levels')
    parser.add_argument('--max-levels', type=int, default=10, help='Max real levels to test')
    parser.add_argument('--timeout', type=float, default=10.0, help='Timeout per algorithm (seconds)')
    parser.add_argument('--no-creative', action='store_true', help='Skip creative algorithms')
    parser.add_argument('--no-weird', action='store_true', help='Skip weird algorithms')
    parser.add_argument('--workers', type=int, help='Number of worker threads')
    
    args = parser.parse_args()
    
    # Load test cases
    if args.real:
        try:
            # Add ftlib path
            ftlib_test_path = os.path.join(os.path.dirname(__file__), '..', 'ftlib', 'test')
            sys.path.append(ftlib_test_path)
            
            from waypoint_test_runner import load_maze_like_levels, create_test_screenshot
            from get_design import retrieveLevel, designDomToStruct
            
            # Load real levels
            tsv_path = os.path.join(os.path.dirname(__file__), '..', 'maze_like_levels.tsv')
            level_ids = load_maze_like_levels(tsv_path)[:args.max_levels]
            
            test_cases = []
            for level_id in level_ids:
                try:
                    level_dom = retrieveLevel(level_id, is_design=False)
                    design_struct = designDomToStruct(level_dom)
                    screenshot = create_test_screenshot(design_struct, (200, 145))
                    test_cases.append((f"real_{level_id}", screenshot))
                except Exception as e:
                    print(f"Failed to load level {level_id}: {e}")
            
            print(f"Loaded {len(test_cases)} real levels")
            
        except ImportError:
            print("Error: ftlib not available for real level testing")
            return 1
    else:
        # Use synthetic test cases
        from waypoint_test_runner import create_synthetic_test_cases
        test_cases = create_synthetic_test_cases()
        print(f"Using {len(test_cases)} synthetic test cases")
    
    if not test_cases:
        print("Error: No test cases available")
        return 1
    
    # Run comprehensive tournament
    results = run_comprehensive_tournament_analysis(
        test_cases=test_cases,
        include_creative=not args.no_creative,
        include_weird=not args.no_weird,
        timeout_per_algorithm=args.timeout,
        save_results=True
    )
    
    # Print summary
    print("\n" + "="*80)
    print("TOURNAMENT SUMMARY")
    print("="*80)
    
    exec_summary = results['execution_summary']
    print(f"Completed {exec_summary['total_tasks']} tasks in {exec_summary['total_time_seconds']:.2f}s")
    print(f"Success rate: {exec_summary['successful_tasks']}/{exec_summary['total_tasks']} "
          f"({100 * exec_summary['successful_tasks']/exec_summary['total_tasks']:.1f}%)")
    
    if 'category_analysis' in results:
        print(f"\nCategory Performance:")
        for cat_name, cat_data in results['category_analysis'].items():
            print(f"  {cat_name.capitalize()}: {cat_data['successful_count']}/{cat_data['algorithm_count']} "
                  f"algorithms successful ({cat_data['success_rate']*100:.1f}%)")
            if cat_data['best_algorithm']:
                print(f"    Best: {cat_data['best_algorithm']} (score: {cat_data['avg_category_score']:.2f})")
    
    if 'surprise_findings' in results and results['surprise_findings']:
        print(f"\n🎉 Surprise findings - Weird algorithms that beat basic ones:")
        for finding in results['surprise_findings']:
            print(f"  {finding['algorithm']}: {finding['score']:.2f} "
                  f"(improved by {finding['improvement']:.2f})")
    
    return 0


if __name__ == "__main__":
    exit(main())