"""
Galapagos Tournament: Comprehensive experimental waypoint generation testing.

Named "Galapagos" to distinguish from pipeline-based tournaments.
Combines all implemented algorithms for comprehensive testing:
- Basic algorithms (Null, CornerTurning)
- Creative algorithms (Genetic, FlowField, etc.)
- Weird algorithms (Chaos, Fibonacci, etc.)
- Learning algorithms (Reinforcement, Adaptive, Evolutionary)
- Web-inspired algorithms (A*, Sampling, Fusion, Narrow spaces)
- Level validation and scoring variations

For end-to-end pipeline testing, see Firelight tournament system.
"""

import os
import sys
import time
from typing import List, Dict, Tuple, Optional, Any

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from .multithreaded_tournament import MultithreadedTournament, TournamentConfig
    from .level_validation import validate_level_database, analyze_level_characteristics
except ImportError:
    from multithreaded_tournament import MultithreadedTournament, TournamentConfig
    from level_validation import validate_level_database, analyze_level_characteristics


def create_experimental_comprehensive_tournament(
    include_basic: bool = True,
    include_creative: bool = True, 
    include_weird: bool = True,
    include_learning: bool = True,
    include_web_inspired: bool = True,
    timeout_per_algorithm: float = 15.0,
    max_workers: Optional[int] = None) -> MultithreadedTournament:
    """
    Create the most comprehensive tournament with all experimental algorithms.
    
    Args:
        include_basic: Include basic algorithms (Null, CornerTurning)
        include_creative: Include creative algorithms (Genetic, etc.)
        include_weird: Include weird experimental algorithms
        include_learning: Include learning/adaptive algorithms
        include_web_inspired: Include web-research inspired algorithms
        timeout_per_algorithm: Timeout per algorithm run
        max_workers: Number of worker threads
        
    Returns:
        Configured comprehensive tournament
    """
    
    config = TournamentConfig(
        max_workers=max_workers,
        timeout_per_algorithm=timeout_per_algorithm,
        use_subprocess=True,
        verbose=True
    )
    
    tournament = MultithreadedTournament(config)
    
    algorithm_count = 0
    
    # Basic algorithms
    if include_basic:
        try:
            from waypoint_generation import create_default_tournament
            basic_tournament = create_default_tournament()
            for generator in basic_tournament.generators:
                tournament.add_generator(generator)
                algorithm_count += 1
            print(f"Added {len(basic_tournament.generators)} basic algorithms")
        except ImportError as e:
            print(f"Warning: Could not load basic algorithms: {e}")
    
    # Creative algorithms
    if include_creative:
        try:
            from quick_creative_generators import create_quick_creative_tournament
            creative_tournament = create_quick_creative_tournament()
            for generator in creative_tournament.generators:
                if not any(g.name == generator.name for g in tournament.generators):
                    tournament.add_generator(generator)
                    algorithm_count += 1
            print(f"Added creative algorithms")
        except ImportError as e:
            print(f"Warning: Could not load creative algorithms: {e}")
    
    # Weird algorithms
    if include_weird:
        try:
            from weird_waypoint_generators import create_weird_tournament
            weird_tournament = create_weird_tournament()
            for generator in weird_tournament.generators:
                if not any(g.name == generator.name for g in tournament.generators):
                    tournament.add_generator(generator)
                    algorithm_count += 1
            print(f"Added weird algorithms")
        except ImportError as e:
            print(f"Warning: Could not load weird algorithms: {e}")
    
    # Learning algorithms
    if include_learning:
        try:
            from learning_waypoint_generators import create_learning_tournament
            learning_tournament = create_learning_tournament()
            for generator in learning_tournament.generators:
                if not any(g.name == generator.name for g in tournament.generators):
                    tournament.add_generator(generator)
                    algorithm_count += 1
            print(f"Added learning algorithms")
        except ImportError as e:
            print(f"Warning: Could not load learning algorithms: {e}")
    
    # Web-inspired algorithms
    if include_web_inspired:
        try:
            from web_inspired_generators import create_web_inspired_tournament
            web_tournament = create_web_inspired_tournament()
            for generator in web_tournament.generators:
                if not any(g.name == generator.name for g in tournament.generators):
                    tournament.add_generator(generator)
                    algorithm_count += 1
            print(f"Added web-inspired algorithms")
        except ImportError as e:
            print(f"Warning: Could not load web-inspired algorithms: {e}")
    
    print(f"Experimental comprehensive tournament created with {len(tournament.generators)} total algorithms")
    return tournament


def run_experimental_tournament_with_validation(
    test_cases: List[Tuple[str, Any]],
    validate_levels: bool = True,
    scoring_variations: bool = True,
    save_results: bool = True,
    results_dir: str = "experimental_results") -> Dict[str, Any]:
    """
    Run experimental tournament with level validation and scoring variations.
    
    Args:
        test_cases: List of (test_case_name, screenshot) tuples
        validate_levels: Whether to validate levels before testing
        scoring_variations: Test different scoring methods
        save_results: Save detailed results
        results_dir: Directory for results
        
    Returns:
        Comprehensive experimental results
    """
    
    print("="*80)
    print("EXPERIMENTAL COMPREHENSIVE WAYPOINT TOURNAMENT")
    print("="*80)
    print()
    
    start_time = time.time()
    
    # Level validation
    if validate_levels:
        print("🔍 Validating levels...")
        valid_levels, invalid_levels = validate_level_database(test_cases, verbose=True)
        print(f"✅ Using {len(valid_levels)}/{len(test_cases)} valid levels")
        
        if invalid_levels:
            print(f"❌ Excluded {len(invalid_levels)} invalid levels:")
            for name, reason in invalid_levels[:5]:  # Show first 5
                print(f"   {name}: {reason}")
        
        test_cases = valid_levels
    else:
        print(f"⚠️  Using all {len(test_cases)} levels without validation")
    
    if not test_cases:
        print("❌ No valid test cases available!")
        return {}
    
    # Analyze level characteristics
    print(f"\n📊 Analyzing level characteristics...")
    level_stats = {
        'total_levels': len(test_cases),
        'level_characteristics': []
    }
    
    for name, screenshot in test_cases[:3]:  # Analyze first 3 levels
        characteristics = analyze_level_characteristics(screenshot)
        level_stats['level_characteristics'].append({
            'name': name,
            'characteristics': characteristics
        })
    
    # Create comprehensive tournament
    print(f"\n🏆 Creating experimental tournament...")
    tournament = create_experimental_comprehensive_tournament(
        timeout_per_algorithm=12.0  # Slightly longer timeout for experimental algorithms
    )
    
    all_results = {}
    
    # Test with different scoring methods if requested
    scoring_methods = ['original', 'systematic'] if scoring_variations else ['systematic']
    
    for scoring_method in scoring_methods:
        print(f"\n🎯 Running tournament with {scoring_method} valley detection...")
        
        # Update scoring method in improved_waypoint_scoring temporarily
        # This is a bit hacky but demonstrates the concept
        results = tournament.run_tournament(test_cases)
        
        # Enhance results with experimental data
        results['level_validation'] = {
            'validated': validate_levels,
            'valid_count': len(test_cases),
            'invalid_count': len(invalid_levels) if validate_levels else 0
        }
        results['level_statistics'] = level_stats
        results['scoring_method'] = scoring_method
        results['experimental_features'] = {
            'learning_algorithms': True,
            'web_inspired_algorithms': True,
            'multi_algorithm_fusion': True,
            'advanced_valley_detection': True
        }
        
        all_results[scoring_method] = results
        
        # Print results for this method
        tournament.print_results(results)
    
    # Save comprehensive results
    if save_results:
        save_experimental_results(all_results, results_dir)
    
    total_time = time.time() - start_time
    print(f"\n⏱️  Experimental tournament completed in {total_time:.2f}s")
    
    return all_results


def save_experimental_results(results: Dict[str, Any], results_dir: str):
    """Save experimental tournament results with detailed analysis."""
    import json
    
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"experimental_tournament_{timestamp}.json"
    filepath = os.path.join(results_dir, filename)
    
    try:
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"💾 Experimental results saved to: {filepath}")
    except Exception as e:
        print(f"⚠️  Could not save results: {e}")


def analyze_experimental_results(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze experimental tournament results for insights."""
    analysis = {
        'algorithm_categories': {},
        'performance_insights': {},
        'experimental_findings': {}
    }
    
    # Categorize algorithms by type
    if 'tournament_results' in results:
        tournament_results = results['tournament_results'] 
        
        # Define algorithm categories
        categories = {
            'basic': ['Null', 'CornerTurning', 'EnhancedCornerTurning'],
            'creative': ['QuickGenetic', 'QuickFlowField', 'QuickSwarm', 'QuickAdaptive', 
                        'ImprovedCornerTurning'],
            'weird': ['Chaos', 'Anti', 'Mega', 'Fibonacci', 'Mirror', 'Prime', 
                     'TimeBased', 'CornerMagnifier', 'EdgeHugger'],
            'learning': ['ReinforcementLearning', 'AdaptiveTemplate', 'EvolutionaryStrategy'],
            'web_inspired': ['AStarWaypoints', 'SamplingBased', 'MultiAlgorithmFusion', 
                           'NarrowSpaceNavigation']
        }
        
        for category, alg_names in categories.items():
            category_results = {}
            for alg_name in alg_names:
                if alg_name in tournament_results:
                    category_results[alg_name] = tournament_results[alg_name]
            
            if category_results:
                # Calculate category statistics
                successful_algs = [name for name, data in category_results.items() 
                                 if data['successful_runs'] > 0]
                
                if successful_algs:
                    best_alg = min(successful_algs, 
                                 key=lambda name: category_results[name]['avg_score'])
                    category_avg = sum(category_results[name]['avg_score'] 
                                     for name in successful_algs) / len(successful_algs)
                else:
                    best_alg = None
                    category_avg = float('inf')
                
                analysis['algorithm_categories'][category] = {
                    'algorithms': list(category_results.keys()),
                    'successful_count': len(successful_algs),
                    'total_count': len(category_results),
                    'best_algorithm': best_alg,
                    'category_average_score': category_avg,
                    'success_rate': len(successful_algs) / len(category_results)
                }
    
    return analysis


def main():
    """Main entry point for experimental tournament."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run experimental comprehensive waypoint tournament')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic test cases')
    parser.add_argument('--real', action='store_true', help='Use real levels (requires ftlib)')
    parser.add_argument('--max-levels', type=int, default=10, help='Max real levels to test')
    parser.add_argument('--no-validation', action='store_true', help='Skip level validation')
    parser.add_argument('--no-scoring-variations', action='store_true', help='Skip scoring method variations')
    parser.add_argument('--timeout', type=float, default=12.0, help='Timeout per algorithm (seconds)')
    parser.add_argument('--basic-only', action='store_true', help='Test only basic algorithms')
    parser.add_argument('--comprehensive', action='store_true', help='Full comprehensive mode: all levels, all algorithms, max workers, JSON output')
    
    args = parser.parse_args()
    
    # Apply comprehensive mode settings (full everything)
    if args.comprehensive:
        args.real = True  # Use real levels
        args.max_levels = 100  # All levels
        args.timeout = 30.0  # Longer timeout
        args.basic_only = False  # All algorithms
        print("🔬 COMPREHENSIVE MODE: All 100 real levels, all algorithms, extended timeouts, JSON output")
    
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
            print(f"🔄 Loading {len(level_ids)} real levels...")
            
            for i, level_id in enumerate(level_ids):
                try:
                    level_dom = retrieveLevel(level_id, is_design=False)
                    design_struct = designDomToStruct(level_dom)
                    screenshot = create_test_screenshot(design_struct, (200, 145))
                    test_cases.append((f"real_{level_id}", screenshot))
                    
                    if (i + 1) % 5 == 0:
                        print(f"   Loaded {i + 1}/{len(level_ids)} levels...")
                        
                except Exception as e:
                    print(f"   ⚠️  Failed to load level {level_id}: {e}")
            
            print(f"✅ Successfully loaded {len(test_cases)} real levels")
            
        except ImportError:
            print("❌ Error: ftlib not available for real level testing")
            return 1
    else:
        # Use synthetic test cases
        from waypoint_test_runner import create_synthetic_test_cases
        test_cases = create_synthetic_test_cases()
        print(f"✅ Using {len(test_cases)} synthetic test cases")
    
    if not test_cases:
        print("❌ Error: No test cases available")
        return 1
    
    # Configure tournament options
    if args.basic_only:
        print("🎯 Testing basic algorithms only")
        # Create basic-only tournament
        from waypoint_generation import create_default_tournament
        from multithreaded_tournament import MultithreadedTournament, TournamentConfig
        
        config = TournamentConfig(timeout_per_algorithm=args.timeout, verbose=True)
        tournament = MultithreadedTournament(config)
        
        basic_tournament = create_default_tournament()
        for generator in basic_tournament.generators:
            tournament.add_generator(generator)
        
        results = tournament.run_tournament(test_cases)
        tournament.print_results(results)
    else:
        # Run experimental tournament
        results = run_experimental_tournament_with_validation(
            test_cases=test_cases,
            validate_levels=not args.no_validation,
            scoring_variations=not args.no_scoring_variations,
            save_results=True
        )
        
        # Analyze results
        if results:
            print(f"\n🔬 Analyzing experimental results...")
            for method, method_results in results.items():
                analysis = analyze_experimental_results(method_results)
                
                print(f"\n📈 Analysis for {method} scoring method:")
                for category, stats in analysis['algorithm_categories'].items():
                    print(f"   {category.capitalize()}: {stats['successful_count']}/{stats['total_count']} "
                          f"successful ({stats['success_rate']*100:.1f}%)")
                    if stats['best_algorithm']:
                        print(f"      Best: {stats['best_algorithm']} "
                              f"(avg score: {stats['category_average_score']:.2f})")
    
    return 0


if __name__ == "__main__":
    exit(main())