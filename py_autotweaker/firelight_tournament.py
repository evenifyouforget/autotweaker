"""
Firelight Tournament: End-to-end pipeline waypoint testing system.

Named "Firelight" to distinguish from Galapagos (waypoint scoring) tournaments.
Tests waypoint lists through the full autotweaker pipeline by:
1. Having algorithms generate waypoints for specific design challenges
2. Running the complete autotweaker workflow with each waypoint list
3. Measuring time-to-solve with statistical analysis across multiple runs
4. Handling timeouts and randomness with proper statistical methods

This provides real-world validation of waypoint generation algorithms.
"""

import os
import sys
import json
import time
import statistics
import threading
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from .waypoint_generation import create_default_tournament
    from .quick_creative_generators import create_quick_creative_tournament
    from .screenshot import screenshot_design
    from get_design import retrieveDesign, designDomToStruct
except ImportError:
    from waypoint_generation import create_default_tournament
    from quick_creative_generators import create_quick_creative_tournament
    from screenshot import screenshot_design
    from get_design import retrieveDesign, designDomToStruct


def normalize_screenshot_colors(screenshot: np.ndarray) -> np.ndarray:
    """
    Normalize screenshot colors to expected convention:
    - 1 = wall (black/dark)
    - 3 = source (blue/cyan)  
    - 4 = sink (red/magenta)
    - 0 = passable air (everything else)
    """
    if len(screenshot.shape) == 3:
        # Convert RGB to single channel classification
        height, width = screenshot.shape[:2]
        normalized = np.zeros((height, width), dtype=np.int32)
        
        # Convert to RGB values
        r, g, b = screenshot[:, :, 0], screenshot[:, :, 1], screenshot[:, :, 2]
        
        # Grayscale for wall detection
        grayscale = np.mean(screenshot, axis=2)
        
        # Dark pixels -> walls (black or very dark)
        wall_mask = grayscale < 50
        normalized[wall_mask] = 1
        
        # Blue-ish pixels -> sources
        # Blue component significantly higher than red/green
        blue_threshold = 200  # Adjust based on actual colors
        blue_mask = (b > blue_threshold) & (b > r + 30) & (b > g + 30) & ~wall_mask
        normalized[blue_mask] = 3
        
        # Red-ish pixels -> sinks
        # Red component significantly higher than blue/green
        red_threshold = 200
        red_mask = (r > red_threshold) & (r > b + 30) & (r > g + 30) & ~wall_mask
        normalized[red_mask] = 4
        
        # Everything else is passable air (0)
        
    else:
        # Already single channel
        normalized = screenshot.astype(np.int32)
        # Ensure only expected values exist
        normalized[(normalized != 1) & (normalized != 3) & (normalized != 4)] = 0
    
    return normalized


class FirelightContestant:
    """Represents a waypoint generation algorithm in Firelight tournament."""
    
    def __init__(self, name: str, waypoints: List[Dict[str, float]], source: str = "algorithm"):
        self.name = name
        self.waypoints = waypoints
        self.source = source  # "algorithm" or "handcrafted"
        self.runs = []
        self.statistics = {}
    
    def add_run_result(self, time_to_solve: Optional[float], timeout_reached: bool, final_score: float):
        """Add result from a single autotweaker run."""
        self.runs.append({
            'time_to_solve': time_to_solve,
            'timeout_reached': timeout_reached,
            'final_score': final_score,
            'success': time_to_solve is not None and not timeout_reached and final_score <= 0
        })
    
    def calculate_statistics(self):
        """Calculate statistical metrics for this contestant."""
        if not self.runs:
            return
        
        success_count = sum(1 for run in self.runs if run['success'])
        timeout_count = sum(1 for run in self.runs if run['timeout_reached'])
        
        # Times to solve (excluding timeouts and failures)
        solve_times = [run['time_to_solve'] for run in self.runs if run['success']]
        
        # Final scores
        final_scores = [run['final_score'] for run in self.runs]
        
        self.statistics = {
            'total_runs': len(self.runs),
            'successes': success_count,
            'timeouts': timeout_count,
            'failures': len(self.runs) - success_count - timeout_count,
            'success_rate': success_count / len(self.runs) if self.runs else 0,
            
            # Time statistics (for successful runs only)
            'avg_solve_time': statistics.mean(solve_times) if solve_times else None,
            'median_solve_time': statistics.median(solve_times) if solve_times else None,
            'min_solve_time': min(solve_times) if solve_times else None,
            'max_solve_time': max(solve_times) if solve_times else None,
            'solve_time_stdev': statistics.stdev(solve_times) if len(solve_times) > 1 else 0,
            
            # Score statistics
            'avg_final_score': statistics.mean(final_scores) if final_scores else float('inf'),
            'best_final_score': min(final_scores) if final_scores else float('inf'),
            'score_stdev': statistics.stdev(final_scores) if len(final_scores) > 1 and all(isinstance(x, (int, float)) and not np.isinf(x) for x in final_scores) else 0.0
        }


class FirelightTournament:
    """
    End-to-end waypoint testing tournament using actual autotweaker pipeline.
    """
    
    def __init__(self, 
                 design_id: int,
                 runs_per_contestant: int = 10,
                 timeout_per_run: int = 300,
                 max_workers: int = 4,
                 screenshot_dimensions: Tuple[int, int] = (400, 300)):
        self.design_id = design_id
        self.runs_per_contestant = runs_per_contestant
        self.timeout_per_run = timeout_per_run
        self.max_workers = max_workers
        self.screenshot_dimensions = screenshot_dimensions
        
        self.contestants = []
        self.design_struct = None
        self.screenshot = None
        
        self._lock = threading.Lock()
        self._load_design()
    
    def _load_design(self):
        """Load design and generate screenshot."""
        print(f"Loading design {self.design_id}...")
        design_dom = retrieveDesign(self.design_id)
        self.design_struct = designDomToStruct(design_dom)
        
        # Generate screenshot
        raw_screenshot = screenshot_design(self.design_struct, self.screenshot_dimensions, use_rgb=True)
        self.screenshot = normalize_screenshot_colors(raw_screenshot)
        
        print(f"Screenshot generated: {self.screenshot.shape}")
        print(f"Pixel value distribution: {np.bincount(self.screenshot.flatten())}")
    
    def add_handcrafted_contestant(self, config_path: str):
        """Add handcrafted waypoints from config file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        waypoints = config.get('waypoints', [])
        contestant = FirelightContestant("Handcrafted", waypoints, source="handcrafted")
        self.contestants.append(contestant)
        print(f"Added handcrafted contestant with {len(waypoints)} waypoints")
    
    def add_algorithm_contestants(self, algorithm_names: Optional[List[str]] = None):
        """Generate waypoint lists from working algorithms."""
        print("Generating waypoint contestants...")
        
        # Load working algorithms from Galapagos tournament results
        working_algorithms = []
        
        # Basic algorithms
        try:
            basic_tournament = create_default_tournament()
            for generator in basic_tournament.generators:
                if algorithm_names is None or generator.name in algorithm_names:
                    working_algorithms.append(generator)
        except Exception as e:
            print(f"Warning: Could not load basic algorithms: {e}")
        
        # Creative algorithms (only successful ones)
        try:
            creative_tournament = create_quick_creative_tournament()
            successful_creative = ['QuickFlowField', 'QuickGenetic', 'ImprovedCornerTurning']
            for generator in creative_tournament.generators:
                if generator.name in successful_creative and (algorithm_names is None or generator.name in algorithm_names):
                    working_algorithms.append(generator)
        except Exception as e:
            print(f"Warning: Could not load creative algorithms: {e}")
        
        # Generate waypoints for each algorithm
        for generator in working_algorithms:
            try:
                print(f"Generating waypoints for {generator.name}...")
                waypoints = generator.generate_waypoints(self.screenshot)
                contestant = FirelightContestant(generator.name, waypoints, source="algorithm")
                self.contestants.append(contestant)
                print(f"  Generated {len(waypoints)} waypoints")
            except Exception as e:
                print(f"  Failed to generate waypoints for {generator.name}: {e}")
        
        print(f"Added {len([c for c in self.contestants if c.source == 'algorithm'])} algorithm contestants")
    
    def _run_single_autotweaker(self, contestant: FirelightContestant, run_id: int) -> Dict[str, Any]:
        """Run single autotweaker instance with given waypoints."""
        run_start_time = time.time()
        
        try:
            # Create temporary config file
            temp_config = {
                "waypoints": contestant.waypoints,
                "tickMinimum": 600,
                "tickMaximum": 4000
            }
            
            temp_config_path = f"/tmp/firelight_config_{contestant.name}_{run_id}_{os.getpid()}.json"
            with open(temp_config_path, 'w') as f:
                json.dump(temp_config, f, indent=2)
            
            # Build command
            autotweaker_dir = Path(__file__).parent.parent
            cmd = [
                './run_local.sh',
                'local',
                '-d', str(self.design_id),
                '-c', temp_config_path,
                '-a',  # --do-autotweak
                '-t', str(self.timeout_per_run),
                '-n', '1',  # Single thread per run for consistency
                '-w',  # --stop-on-win
                '-k', '0'  # Don't upload during tournament
            ]
            
            # Run autotweaker with timeout
            process = subprocess.run(
                cmd,
                cwd=autotweaker_dir,
                capture_output=True,
                text=True,
                timeout=self.timeout_per_run + 30  # Extra buffer for process overhead
            )
            
            # Cleanup
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
            
            # Parse results from output
            output = process.stdout
            timeout_reached = "time limit reached" in output
            
            # Extract final score (simplified parsing)
            final_score = float('inf')
            time_to_solve = None
            
            # Look for score indicators in output
            for line in output.split('\n'):
                if 'Best score so far:' in line:
                    try:
                        score = float(line.split('Best score so far:')[1].strip())
                        final_score = min(final_score, score)
                    except (ValueError, IndexError):
                        pass
                
                if 'Stopping early since a solve was found' in line:
                    time_to_solve = time.time() - run_start_time
            
            if time_to_solve is None and not timeout_reached and final_score <= 0:
                # Solved but didn't catch the exact moment
                time_to_solve = time.time() - run_start_time
            
            return {
                'time_to_solve': time_to_solve,
                'timeout_reached': timeout_reached,
                'final_score': final_score,
                'stdout': output,
                'stderr': process.stderr,
                'return_code': process.returncode
            }
            
        except subprocess.TimeoutExpired:
            # Cleanup
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
            
            return {
                'time_to_solve': None,
                'timeout_reached': True,
                'final_score': float('inf'),
                'stdout': '',
                'stderr': 'Process timeout',
                'return_code': -1
            }
        
        except Exception as e:
            # Cleanup
            if 'temp_config_path' in locals() and os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
            
            return {
                'time_to_solve': None,
                'timeout_reached': False,
                'final_score': float('inf'),
                'stdout': '',
                'stderr': str(e),
                'return_code': -2
            }
    
    def run_tournament(self, verbose: bool = True) -> Dict[str, Any]:
        """Run the complete Firelight tournament."""
        if not self.contestants:
            raise ValueError("No contestants added to tournament")
        
        start_time = time.time()
        total_runs = len(self.contestants) * self.runs_per_contestant
        completed_runs = 0
        
        print("=" * 80)
        print("FIRELIGHT TOURNAMENT: End-to-End Waypoint Pipeline Testing")
        print("=" * 80)
        print(f"Design ID: {self.design_id}")
        print(f"Contestants: {len(self.contestants)}")
        print(f"Runs per contestant: {self.runs_per_contestant}")
        print(f"Timeout per run: {self.timeout_per_run}s")
        print(f"Total runs: {total_runs}")
        print(f"Max workers: {self.max_workers}")
        print()
        
        # Create tasks for all runs
        tasks = []
        for contestant in self.contestants:
            for run_id in range(self.runs_per_contestant):
                tasks.append((contestant, run_id))
        
        # Execute all runs with thread pool
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._run_single_autotweaker, contestant, run_id): (contestant, run_id)
                for contestant, run_id in tasks
            }
            
            # Process completed tasks
            for future in as_completed(future_to_task):
                contestant, run_id = future_to_task[future]
                
                try:
                    result = future.result()
                    contestant.add_run_result(
                        result['time_to_solve'],
                        result['timeout_reached'],
                        result['final_score']
                    )
                    
                    completed_runs += 1
                    
                    if verbose:
                        status = "SOLVED" if result['time_to_solve'] is not None and result['final_score'] <= 0 else \
                                "TIMEOUT" if result['timeout_reached'] else \
                                f"SCORE:{result['final_score']:.1f}"
                        
                        elapsed = time.time() - start_time
                        print(f"[{completed_runs:3d}/{total_runs:3d}] {contestant.name:20} Run {run_id+1:2d}: {status:12} "
                              f"({elapsed:6.1f}s elapsed)")
                
                except Exception as e:
                    print(f"Error processing run for {contestant.name}: {e}")
                    completed_runs += 1
        
        # Calculate statistics for all contestants
        for contestant in self.contestants:
            contestant.calculate_statistics()
        
        # Prepare results
        results = {
            'tournament_info': {
                'design_id': self.design_id,
                'total_runs': total_runs,
                'runs_per_contestant': self.runs_per_contestant,
                'timeout_per_run': self.timeout_per_run,
                'total_time': time.time() - start_time
            },
            'contestants': self.contestants
        }
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted tournament results."""
        print("\n" + "=" * 80)
        print("FIRELIGHT TOURNAMENT RESULTS")
        print("=" * 80)
        
        info = results['tournament_info']
        print(f"Design ID: {info['design_id']}")
        print(f"Total execution time: {info['total_time']:.1f}s")
        print(f"Total runs: {info['total_runs']} ({info['runs_per_contestant']} per contestant)")
        print()
        
        # Sort contestants by performance (success rate, then average solve time)
        contestants = sorted(results['contestants'], 
                           key=lambda c: (-c.statistics['success_rate'], 
                                        c.statistics['avg_solve_time'] or float('inf')))
        
        print("Contestant Performance:")
        print("-" * 80)
        print(f"{'Rank':<4} {'Name':<20} {'Source':<12} {'Success':<8} {'Avg Time':<10} {'Best Score':<12}")
        print("-" * 80)
        
        for i, contestant in enumerate(contestants, 1):
            stats = contestant.statistics
            
            success_str = f"{stats['successes']}/{stats['total_runs']}"
            success_pct = f"({stats['success_rate']*100:.1f}%)"
            
            avg_time_str = f"{stats['avg_solve_time']:.1f}s" if stats['avg_solve_time'] else "N/A"
            if stats['avg_solve_time'] and stats['solve_time_stdev'] > 0:
                avg_time_str += f"±{stats['solve_time_stdev']:.1f}"
            
            best_score_str = f"{stats['best_final_score']:.1f}" if stats['best_final_score'] != float('inf') else "∞"
            
            print(f"{i:<4} {contestant.name:<20} {contestant.source:<12} "
                  f"{success_str:<8} {avg_time_str:<10} {best_score_str:<12}")
        
        print("-" * 80)
        
        # Detailed statistics
        if any(c.statistics['successes'] > 0 for c in contestants):
            print("\nDetailed Statistics (Successful Contestants Only):")
            print("-" * 80)
            
            for contestant in contestants:
                if contestant.statistics['successes'] == 0:
                    continue
                
                stats = contestant.statistics
                print(f"\n{contestant.name} ({contestant.source}):")
                print(f"  Success rate: {stats['success_rate']*100:.1f}% ({stats['successes']}/{stats['total_runs']} runs)")
                
                if stats['avg_solve_time']:
                    print(f"  Solve time: {stats['avg_solve_time']:.1f}s avg, {stats['median_solve_time']:.1f}s median")
                    print(f"              {stats['min_solve_time']:.1f}s min, {stats['max_solve_time']:.1f}s max")
                    if stats['solve_time_stdev'] > 0:
                        print(f"              {stats['solve_time_stdev']:.1f}s std dev")
                
                print(f"  Final score: {stats['best_final_score']:.1f} best, {stats['avg_final_score']:.1f} avg")
                print(f"  Timeouts: {stats['timeouts']}, Failures: {stats['failures']}")
        
        print("=" * 80)


def create_firelight_tournament(design_id: int, **kwargs) -> FirelightTournament:
    """Create a Firelight tournament for the given design."""
    return FirelightTournament(design_id, **kwargs)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run Firelight end-to-end waypoint tournament')
    parser.add_argument('design_id', type=int, help='Design ID to test')
    parser.add_argument('--runs', type=int, default=3, help='Runs per contestant (default: 3)')
    parser.add_argument('--timeout', type=int, default=120, help='Timeout per run in seconds (default: 120)')
    parser.add_argument('--workers', type=int, default=2, help='Max parallel workers (default: 2)')
    parser.add_argument('--handcrafted-config', type=str, help='Path to handcrafted config file')
    parser.add_argument('--algorithms', nargs='*', help='Specific algorithms to test')
    parser.add_argument('--quick', action='store_true', help='Quick test with minimal runs')
    
    args = parser.parse_args()
    
    # Adjust settings for quick test
    if args.quick:
        args.runs = min(args.runs, 2)
        args.timeout = min(args.timeout, 60)
    
    # Create tournament
    tournament = create_firelight_tournament(
        design_id=args.design_id,
        runs_per_contestant=args.runs,
        timeout_per_run=args.timeout,
        max_workers=args.workers
    )
    
    # Add handcrafted contestant if specified
    if args.handcrafted_config:
        tournament.add_handcrafted_contestant(args.handcrafted_config)
    
    # Add algorithm contestants
    tournament.add_algorithm_contestants(args.algorithms)
    
    if not tournament.contestants:
        print("No contestants added to tournament!")
        exit(1)
    
    # Run tournament
    results = tournament.run_tournament()
    tournament.print_results(results)
    
    # Save results
    results_dir = Path(__file__).parent / "firelight_results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"firelight_{args.design_id}_{timestamp}.json"
    
    # Convert results to JSON-serializable format
    json_results = {
        'tournament_info': results['tournament_info'],
        'contestants': []
    }
    
    for contestant in results['contestants']:
        json_results['contestants'].append({
            'name': contestant.name,
            'source': contestant.source,
            'waypoints': contestant.waypoints,
            'statistics': contestant.statistics,
            'runs': contestant.runs
        })
    
    with open(results_file, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")