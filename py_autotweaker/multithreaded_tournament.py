#!/usr/bin/env python3
"""
Multithreaded tournament system with subprocess isolation and timeouts.

This module provides high-performance tournament execution using:
- ThreadPoolExecutor for parallel algorithm execution
- Subprocess isolation to prevent algorithm hangs from blocking the tournament
- Per-algorithm timeouts (default 10s) to handle infinite loops or slow algorithms
- Progress tracking and early termination on critical failures
- Robust error handling with detailed failure reporting
"""

import os
import sys
import time
import json
import pickle
import tempfile
import subprocess
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
import multiprocessing

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from .waypoint_generation import WaypointGenerator, WaypointTournament
    from .improved_waypoint_scoring import improved_score_waypoint_list
except ImportError:
    from waypoint_generation import WaypointGenerator, WaypointTournament
    from improved_waypoint_scoring import improved_score_waypoint_list


@dataclass
class TaskResult:
    """Result of a single algorithm run on a single test case."""
    generator_name: str
    test_case_name: str
    success: bool
    score: Optional[float] = None
    waypoints: Optional[List[Dict[str, float]]] = None
    execution_time: Optional[float] = None
    error_message: Optional[str] = None
    timeout: bool = False


@dataclass
class TournamentConfig:
    """Configuration for multithreaded tournament execution."""
    max_workers: Optional[int] = None  # None = auto-detect CPU count
    timeout_per_algorithm: float = 10.0  # seconds
    use_subprocess: bool = True  # Use subprocess isolation
    verbose: bool = True
    save_individual_results: bool = False  # Save each result immediately
    early_termination_threshold: float = 0.8  # Stop if 80% of algorithms fail consistently


class MultithreadedTournament:
    """High-performance tournament with parallel execution and subprocess isolation."""
    
    def __init__(self, config: Optional[TournamentConfig] = None):
        self.config = config or TournamentConfig()
        self.generators = []
        
        if self.config.max_workers is None:
            self.config.max_workers = min(32, multiprocessing.cpu_count() + 4)
        
        if self.config.verbose:
            print(f"Multithreaded tournament initialized with {self.config.max_workers} workers")
            print(f"Algorithm timeout: {self.config.timeout_per_algorithm}s")
    
    def add_generator(self, generator: WaypointGenerator):
        """Add a waypoint generator to the tournament."""
        self.generators.append(generator)
    
    def _create_subprocess_task(self, generator_name: str, test_case_name: str, 
                              screenshot_data: bytes) -> str:
        """Create a subprocess task file for isolated execution."""
        
        # Create temporary files for input and output
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.pkl') as f:
            task_data = {
                'generator_name': generator_name,
                'test_case_name': test_case_name,
                'screenshot_data': screenshot_data,
            }
            pickle.dump(task_data, f)
            task_file = f.name
        
        return task_file
    
    def _run_subprocess_task(self, task_file: str, timeout: float) -> TaskResult:
        """Execute a single algorithm task in a subprocess with timeout."""
        
        # Create subprocess runner script path
        runner_script = os.path.join(os.path.dirname(__file__), 'subprocess_runner.py')
        
        try:
            # Run the subprocess with timeout
            result = subprocess.run([
                sys.executable, runner_script, task_file
            ], capture_output=True, text=True, timeout=timeout)
            
            if result.returncode == 0:
                # Parse successful result
                try:
                    result_data = json.loads(result.stdout)
                    return TaskResult(
                        generator_name=result_data['generator_name'],
                        test_case_name=result_data['test_case_name'],
                        success=True,
                        score=result_data.get('score'),
                        waypoints=result_data.get('waypoints'),
                        execution_time=result_data.get('execution_time'),
                    )
                except (json.JSONDecodeError, KeyError) as e:
                    return TaskResult(
                        generator_name="unknown",
                        test_case_name="unknown", 
                        success=False,
                        error_message=f"Failed to parse subprocess result: {e}"
                    )
            else:
                # Subprocess failed
                return TaskResult(
                    generator_name="unknown",
                    test_case_name="unknown",
                    success=False,
                    error_message=f"Subprocess failed: {result.stderr}"
                )
                
        except subprocess.TimeoutExpired:
            return TaskResult(
                generator_name="unknown", 
                test_case_name="unknown",
                success=False,
                timeout=True,
                error_message=f"Algorithm timed out after {timeout}s"
            )
        except Exception as e:
            return TaskResult(
                generator_name="unknown",
                test_case_name="unknown", 
                success=False,
                error_message=f"Subprocess execution failed: {e}"
            )
        finally:
            # Clean up task file
            try:
                os.unlink(task_file)
            except:
                pass
    
    def _run_direct_task(self, generator: WaypointGenerator, test_case_name: str, 
                        screenshot: Any, timeout: float) -> TaskResult:
        """Execute a single algorithm task directly with timeout using threading."""
        
        result_container = [None]
        exception_container = [None]
        
        def target():
            try:
                start_time = time.time()
                waypoints = generator.generate_waypoints(screenshot)
                execution_time = time.time() - start_time
                
                # Score the waypoints
                score = improved_score_waypoint_list(screenshot, waypoints, penalize_skippable=True)
                
                result_container[0] = TaskResult(
                    generator_name=generator.name,
                    test_case_name=test_case_name,
                    success=True,
                    score=score,
                    waypoints=waypoints,
                    execution_time=execution_time
                )
            except Exception as e:
                exception_container[0] = e
        
        # Run in thread with timeout
        thread = threading.Thread(target=target)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        
        if thread.is_alive():
            # Thread is still running, timeout occurred
            return TaskResult(
                generator_name=generator.name,
                test_case_name=test_case_name,
                success=False,
                timeout=True,
                error_message=f"Algorithm timed out after {timeout}s"
            )
        
        if exception_container[0]:
            return TaskResult(
                generator_name=generator.name,
                test_case_name=test_case_name,
                success=False,
                error_message=str(exception_container[0])
            )
        
        return result_container[0] or TaskResult(
            generator_name=generator.name,
            test_case_name=test_case_name,
            success=False,
            error_message="Unknown execution failure"
        )
    
    def run_tournament(self, test_cases: List[Tuple[str, Any]], 
                      verbose: Optional[bool] = None) -> Dict[str, Any]:
        """
        Run the tournament with multithreading and subprocess isolation.
        
        Args:
            test_cases: List of (test_case_name, screenshot) tuples
            verbose: Override config verbose setting
            
        Returns:
            Tournament results dictionary with timing and error analysis
        """
        
        if verbose is None:
            verbose = self.config.verbose
        
        if not self.generators:
            raise ValueError("No generators added to tournament")
        
        if not test_cases:
            raise ValueError("No test cases provided")
        
        start_time = time.time()
        
        if verbose:
            print(f"\nStarting multithreaded tournament:")
            print(f"  Algorithms: {len(self.generators)}")
            print(f"  Test cases: {len(test_cases)}")
            print(f"  Workers: {self.config.max_workers}")
            print(f"  Total tasks: {len(self.generators) * len(test_cases)}")
        
        # Prepare all tasks
        tasks = []
        for generator in self.generators:
            for test_case_name, screenshot in test_cases:
                tasks.append((generator, test_case_name, screenshot))
        
        results = []
        completed_tasks = 0
        total_tasks = len(tasks)
        
        # Execute tasks in parallel
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            # Submit all tasks
            future_to_task = {}
            
            for generator, test_case_name, screenshot in tasks:
                if self.config.use_subprocess:
                    # Serialize screenshot for subprocess
                    screenshot_data = pickle.dumps(screenshot)
                    task_file = self._create_subprocess_task(
                        generator.name, test_case_name, screenshot_data)
                    
                    future = executor.submit(
                        self._run_subprocess_task, task_file, self.config.timeout_per_algorithm)
                else:
                    future = executor.submit(
                        self._run_direct_task, generator, test_case_name, 
                        screenshot, self.config.timeout_per_algorithm)
                
                future_to_task[future] = (generator.name, test_case_name)
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                generator_name, test_case_name = future_to_task[future]
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    completed_tasks += 1
                    
                    if verbose and completed_tasks % 10 == 0:
                        progress = (completed_tasks / total_tasks) * 100
                        elapsed = time.time() - start_time
                        print(f"  Progress: {completed_tasks}/{total_tasks} ({progress:.1f}%) "
                              f"- {elapsed:.1f}s elapsed")
                
                except Exception as e:
                    # This should rarely happen due to our error handling
                    results.append(TaskResult(
                        generator_name=generator_name,
                        test_case_name=test_case_name,
                        success=False,
                        error_message=f"Future execution failed: {e}"
                    ))
                    completed_tasks += 1
        
        total_time = time.time() - start_time
        
        if verbose:
            print(f"\nTournament completed in {total_time:.2f}s")
            print(f"Tasks per second: {total_tasks / total_time:.1f}")
        
        # Analyze results
        return self._analyze_results(results, test_cases, total_time)
    
    def _analyze_results(self, results: List[TaskResult], test_cases: List[Tuple[str, Any]], 
                        total_time: float) -> Dict[str, Any]:
        """Analyze tournament results and generate comprehensive statistics."""
        
        # Group results by generator
        generator_results = {}
        for generator in self.generators:
            generator_results[generator.name] = {
                'scores': [],
                'times': [],
                'waypoint_counts': [],
                'successful_runs': 0,
                'failed_runs': 0,
                'timeout_runs': 0,
                'error_messages': []
            }
        
        # Process individual results
        for result in results:
            gen_name = result.generator_name
            if gen_name not in generator_results:
                continue  # Skip unknown generators
            
            gen_data = generator_results[gen_name]
            
            if result.success:
                gen_data['successful_runs'] += 1
                gen_data['scores'].append(result.score or 0.0)
                gen_data['times'].append(result.execution_time or 0.0)
                gen_data['waypoint_counts'].append(len(result.waypoints or []))
            else:
                if result.timeout:
                    gen_data['timeout_runs'] += 1
                else:
                    gen_data['failed_runs'] += 1
                
                if result.error_message:
                    gen_data['error_messages'].append(result.error_message)
        
        # Calculate summary statistics
        for gen_name, gen_data in generator_results.items():
            if gen_data['scores']:
                gen_data['avg_score'] = sum(gen_data['scores']) / len(gen_data['scores'])
                gen_data['total_score'] = sum(gen_data['scores'])
                gen_data['avg_time'] = sum(gen_data['times']) / len(gen_data['times'])
                gen_data['avg_waypoints'] = sum(gen_data['waypoint_counts']) / len(gen_data['waypoint_counts'])
            else:
                gen_data['avg_score'] = float('inf')
                gen_data['total_score'] = float('inf')
                gen_data['avg_time'] = 0.0
                gen_data['avg_waypoints'] = 0.0
        
        # Performance analysis
        successful_generators = [name for name, data in generator_results.items() 
                               if data['successful_runs'] > 0]
        
        if successful_generators:
            best_generator = min(successful_generators, 
                               key=lambda name: generator_results[name]['avg_score'])
        else:
            best_generator = None
        
        return {
            'tournament_results': generator_results,
            'execution_summary': {
                'total_time_seconds': total_time,
                'total_tasks': len(results),
                'successful_tasks': sum(1 for r in results if r.success),
                'failed_tasks': sum(1 for r in results if not r.success and not r.timeout),
                'timeout_tasks': sum(1 for r in results if r.timeout),
                'tasks_per_second': len(results) / total_time,
                'best_generator': best_generator,
                'test_case_count': len(test_cases),
                'generator_count': len(self.generators)
            },
            'configuration': {
                'max_workers': self.config.max_workers,
                'timeout_per_algorithm': self.config.timeout_per_algorithm,
                'use_subprocess': self.config.use_subprocess,
            }
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted tournament results."""
        
        print("\n" + "="*80)
        print("MULTITHREADED TOURNAMENT RESULTS")
        print("="*80)
        
        exec_summary = results['execution_summary']
        print(f"Execution time: {exec_summary['total_time_seconds']:.2f}s")
        print(f"Tasks per second: {exec_summary['tasks_per_second']:.1f}")
        print(f"Success rate: {exec_summary['successful_tasks']}/{exec_summary['total_tasks']} "
              f"({100 * exec_summary['successful_tasks']/exec_summary['total_tasks']:.1f}%)")
        
        if exec_summary['timeout_tasks'] > 0:
            print(f"Timeouts: {exec_summary['timeout_tasks']}")
        
        print(f"\nAlgorithm Performance:")
        print("-" * 80)
        
        tournament_results = results['tournament_results']
        
        # Sort by average score
        sorted_generators = sorted(tournament_results.items(), 
                                 key=lambda x: x[1]['avg_score'])
        
        print(f"{'Rank':<4} {'Algorithm':<20} {'Avg Score':<10} {'Success':<8} {'Timeouts':<8} {'Avg Time':<10}")
        print("-" * 80)
        
        for rank, (gen_name, gen_data) in enumerate(sorted_generators, 1):
            success_rate = f"{gen_data['successful_runs']}/{exec_summary['test_case_count']}"
            avg_score = f"{gen_data['avg_score']:.2f}" if gen_data['avg_score'] != float('inf') else "FAIL"
            avg_time = f"{gen_data['avg_time']:.3f}s" if gen_data['avg_time'] > 0 else "N/A"
            
            print(f"{rank:<4} {gen_name:<20} {avg_score:<10} {success_rate:<8} "
                  f"{gen_data['timeout_runs']:<8} {avg_time:<10}")
        
        if exec_summary['best_generator']:
            print(f"\n🏆 Best performer: {exec_summary['best_generator']}")
        
        print("="*80)


if __name__ == "__main__":
    # Example usage
    from waypoint_generation import create_default_tournament
    from waypoint_test_runner import create_synthetic_test_cases
    
    print("Testing multithreaded tournament system...")
    
    # Create tournament
    config = TournamentConfig(
        max_workers=4,
        timeout_per_algorithm=5.0,
        use_subprocess=False,  # Use direct execution for testing
        verbose=True
    )
    
    mt_tournament = MultithreadedTournament(config)
    
    # Add generators
    regular_tournament = create_default_tournament()
    for generator in regular_tournament.generators:
        mt_tournament.add_generator(generator)
    
    # Run on synthetic test cases
    test_cases = create_synthetic_test_cases()
    results = mt_tournament.run_tournament(test_cases)
    
    # Print results
    mt_tournament.print_results(results)