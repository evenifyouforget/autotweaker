#!/usr/bin/env python3
"""
Subprocess-based tournament system with isolation and timeouts.

This module provides high-performance tournament execution using:
- Pure subprocess-based parallel execution (no threading, avoids GIL)
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
# Removed threading imports - now using pure subprocess-based parallelism
from typing import List, Dict, Tuple, Optional, Any, Union, Protocol, runtime_checkable
from dataclasses import dataclass
import multiprocessing
import numpy as np

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

try:
    from .improved_waypoint_scoring import improved_score_waypoint_list
except ImportError:
    from improved_waypoint_scoring import improved_score_waypoint_list


@runtime_checkable
class WaypointGeneratorProtocol(Protocol):
    """Protocol for waypoint generators to provide better type checking."""
    name: str
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints from a screenshot."""
        ...


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
    """Configuration for subprocess tournament execution."""
    max_workers: Optional[int] = None  # None = auto-detect CPU count
    timeout_per_algorithm: float = 10.0  # seconds
    verbose: bool = True
    save_individual_results: bool = False  # Save each result immediately
    early_termination_threshold: float = 0.8  # Stop if 80% of algorithms fail consistently


class WaypointTournament:
    """High-performance tournament with subprocess-based parallel execution.
    
    Supports both single-worker (max_workers=1) and multi-worker execution.
    This replaces the old single-threaded WaypointTournament implementation.
    """
    
    def __init__(self, config: Optional[TournamentConfig] = None, feature_flags: Optional[Dict[str, bool]] = None):
        self.config = config or TournamentConfig()
        self.generators: List[WaypointGeneratorProtocol] = []
        self.feature_flags = feature_flags or {}  # Compatibility with old interface
        self.results: Dict[str, Dict[str, Any]] = {}  # Compatibility with old interface
        
        if self.config.max_workers is None:
            self.config.max_workers = min(32, multiprocessing.cpu_count() + 4)
        
        if self.config.verbose:
            workers_desc = "single-threaded" if self.config.max_workers == 1 else f"{self.config.max_workers} workers"
            print(f"Tournament initialized with {workers_desc}")
            print(f"Algorithm timeout: {self.config.timeout_per_algorithm}s")
    
    def add_generator(self, generator: WaypointGeneratorProtocol) -> None:
        """Add a waypoint generator to the tournament."""
        self.generators.append(generator)
    
    def add_generator_class(self, generator_class: type[WaypointGeneratorProtocol], *args: Any, **kwargs: Any) -> None:
        """Add a generator class to the tournament (compatibility method)."""
        generator = generator_class(*args, **kwargs)
        self.add_generator(generator)
    
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
    
    
    def run_tournament(self, test_cases: List[Tuple[str, Union[np.ndarray, Any]]], 
                      verbose: Optional[bool] = None) -> Dict[str, Any]:
        """
        Run the tournament with subprocess-based parallelism.
        
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
            print(f"\nStarting subprocess tournament:")
            print(f"  Algorithms: {len(self.generators)}")
            print(f"  Test cases: {len(test_cases)}")
            print(f"  Workers: {self.config.max_workers}")
            print(f"  Total tasks: {len(self.generators) * len(test_cases)}")
        
        # Prepare all tasks (only subprocess execution now)
        tasks = []
        for generator in self.generators:
            for test_case_name, screenshot in test_cases:
                tasks.append((generator, test_case_name, screenshot))
        
        results = []
        completed_tasks = 0
        total_tasks = len(tasks)
        
        # Execute tasks in parallel using subprocess batching
        # Ensure max_workers is set (fallback if somehow None)
        max_workers = self.config.max_workers
        if max_workers is None:
            max_workers = min(32, multiprocessing.cpu_count() + 4)
        max_parallel = min(len(tasks), max_workers)
        
        for i in range(0, len(tasks), max_parallel):
            batch = tasks[i:i + max_parallel]
            
            # Start all processes in batch
            processes = []
            for generator, test_case_name, screenshot in batch:
                # Serialize screenshot for subprocess
                screenshot_data = pickle.dumps(screenshot)
                task_file = self._create_subprocess_task(
                    generator.name, test_case_name, screenshot_data)
                
                # Start subprocess
                runner_script = os.path.join(os.path.dirname(__file__), 'subprocess_runner.py')
                proc = subprocess.Popen([
                    sys.executable, runner_script, task_file
                ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                
                processes.append((proc, task_file, generator.name, test_case_name))
            
            # Wait for all processes in batch to complete
            for proc, task_file, generator_name, test_case_name in processes:
                try:
                    stdout, stderr = proc.communicate(timeout=self.config.timeout_per_algorithm)
                    
                    if proc.returncode == 0:
                        # Parse successful result
                        try:
                            result_data = json.loads(stdout)
                            result = TaskResult(
                                generator_name=result_data['generator_name'],
                                test_case_name=result_data['test_case_name'],
                                success=True,
                                score=result_data.get('score'),
                                waypoints=result_data.get('waypoints'),
                                execution_time=result_data.get('execution_time'),
                            )
                        except (json.JSONDecodeError, KeyError) as e:
                            result = TaskResult(
                                generator_name=generator_name,
                                test_case_name=test_case_name, 
                                success=False,
                                error_message=f"Failed to parse subprocess result: {e}"
                            )
                    else:
                        # Subprocess failed
                        result = TaskResult(
                            generator_name=generator_name,
                            test_case_name=test_case_name,
                            success=False,
                            error_message=f"Subprocess failed: {stderr}"
                        )
                        
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.communicate()  # Clean up zombie process
                    result = TaskResult(
                        generator_name=generator_name, 
                        test_case_name=test_case_name,
                        success=False,
                        timeout=True,
                        error_message=f"Algorithm timed out after {self.config.timeout_per_algorithm}s"
                    )
                except Exception as e:
                    try:
                        proc.kill()
                        proc.communicate()
                    except:
                        pass
                    result = TaskResult(
                        generator_name=generator_name,
                        test_case_name=test_case_name, 
                        success=False,
                        error_message=f"Subprocess execution failed: {e}"
                    )
                finally:
                    # Clean up task file
                    try:
                        os.unlink(task_file)
                    except:
                        pass
                
                results.append(result)
                completed_tasks += 1
                
                if verbose and completed_tasks % 10 == 0:
                    progress = (completed_tasks / total_tasks) * 100
                    elapsed = time.time() - start_time
                    print(f"  Progress: {completed_tasks}/{total_tasks} ({progress:.1f}%) "
                          f"- {elapsed:.1f}s elapsed")
        
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
            
            # Add compatibility fields for old interface
            gen_data['error_count'] = gen_data['failed_runs'] + gen_data['timeout_runs']
            gen_data['skippable_count'] = 0  # Not tracked in this implementation
        
        # Store results in self.results for compatibility with old interface
        self.results = generator_results
        
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
                'timeout_per_algorithm': self.config.timeout_per_algorithm
            }
        }
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted tournament results."""
        
        print("\n" + "="*80)
        exec_type = "SINGLE-THREADED" if self.config.max_workers == 1 else "PARALLEL"
        print(f"{exec_type} TOURNAMENT RESULTS")
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


def create_default_tournament(max_workers: Optional[int] = None, feature_flags: Optional[Dict[str, bool]] = None) -> 'WaypointTournament':
    """Create a default tournament with basic generators (compatibility function).
    
    Args:
        max_workers: Number of workers (None = auto-detect, 1 = single-threaded)
        feature_flags: Feature flags (for compatibility, currently ignored)
    
    Returns:
        WaypointTournament configured with basic generators
    """
    from waypoint_generation import NullGenerator, CornerTurningGenerator
    
    config = TournamentConfig(
        max_workers=max_workers,
        timeout_per_algorithm=10.0,
        verbose=False
    )
    
    tournament = WaypointTournament(config, feature_flags)
    
    # Add basic generators
    tournament.add_generator(NullGenerator())
    tournament.add_generator(CornerTurningGenerator())
    
    return tournament


if __name__ == "__main__":
    # Example usage
    from waypoint_generation import NullGenerator, CornerTurningGenerator
    from waypoint_test_runner import create_synthetic_test_cases
    
    print("Testing subprocess tournament system...")
    
    # Create tournament
    config = TournamentConfig(
        max_workers=4,
        timeout_per_algorithm=5.0,
        verbose=True
    )
    
    tournament = WaypointTournament(config)
    
    # Add generators manually to avoid circular import
    tournament.add_generator(NullGenerator())
    tournament.add_generator(CornerTurningGenerator())
    
    # Run on synthetic test cases
    test_cases = create_synthetic_test_cases()
    results = tournament.run_tournament(test_cases)
    
    # Print results
    tournament.print_results(results)