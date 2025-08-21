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
import subprocess
import logging
import traceback
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import numpy as np

# Add paths for imports (support running from any directory)
current_dir = os.path.dirname(os.path.abspath(__file__))
autotweaker_root = os.path.dirname(current_dir)
ftlib_test_dir = os.path.join(autotweaker_root, 'ftlib', 'test')

sys.path.insert(0, current_dir)
sys.path.insert(0, autotweaker_root)
sys.path.insert(0, ftlib_test_dir)

try:
    from waypoint_generation import create_default_tournament
    from quick_creative_generators import create_quick_creative_tournament
    from screenshot import screenshot_design
    from coordinate_transform import pixel_to_world, world_to_pixel
    from get_design import retrieveDesign, designDomToStruct
    from improved_waypoint_scoring import improved_score_waypoint_list
    from waypoint_scoring import score_waypoint_list
except ImportError as e:
    print(f"Import error: {e}")
    print(f"Current directory: {current_dir}")
    print(f"Autotweaker root: {autotweaker_root}")
    print(f"ftlib test dir: {ftlib_test_dir}")
    print(f"Python path: {sys.path}")
    raise


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
                 max_workers: Optional[int] = None,
                 screenshot_dimensions: Tuple[int, int] = (400, 290),
                 dry_run: bool = False):
        self.design_id = design_id
        self.runs_per_contestant = runs_per_contestant
        self.timeout_per_run = timeout_per_run
        self.dry_run = dry_run
        
        # Auto-detect max workers if not specified
        if max_workers is None:
            import multiprocessing
            self.max_workers = multiprocessing.cpu_count()
        else:
            self.max_workers = max_workers
            
        self.screenshot_dimensions = screenshot_dimensions
        
        self.contestants = []
        self.design_struct = None
        self.screenshot = None
        self.exception_count = 0
        
        # Setup exception logging
        self._setup_exception_logging()
        
        self._load_design()
    
    def _setup_exception_logging(self):
        """Setup file logging for exceptions."""
        log_dir = Path('firelight_results')
        log_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.exception_log_file = log_dir / f'firelight_exceptions_{self.design_id}_{timestamp}.log'
        
        # Setup logger
        self.exception_logger = logging.getLogger(f'firelight_{self.design_id}')
        self.exception_logger.setLevel(logging.ERROR)
        
        # File handler
        handler = logging.FileHandler(self.exception_log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.exception_logger.addHandler(handler)
    
    def _log_exception(self, context: str, exception: Exception, extra_info: str = ""):
        """Log exception to both console and file."""
        self.exception_count += 1
        
        error_msg = f"[{self.exception_count}] {context}: {str(exception)}"
        full_traceback = traceback.format_exc()
        
        # Log to file
        self.exception_logger.error(f"{error_msg}\n{extra_info}\n{full_traceback}\n{'='*80}")
        
        # Print first exception details to console  
        if self.exception_count == 1:
            print(f"\n❌ FIRST EXCEPTION DETAILS:")
            print(f"   Context: {context}")
            print(f"   Error: {str(exception)}")
            if extra_info:
                print(f"   Extra: {extra_info}")
            print(f"   Full details logged to: {self.exception_log_file}")
            sys.stdout.flush()
        else:
            print(f"❌ Exception #{self.exception_count}: {error_msg[:100]}... (logged to file)")
            sys.stdout.flush()
    
    def _load_design(self):
        """Load design and generate screenshot."""
        print(f"Loading design {self.design_id}...")
        design_dom = retrieveDesign(self.design_id)
        self.design_struct = designDomToStruct(design_dom)
        
        # Generate screenshot
        # Use raw integer screenshot with proper normalization 
        raw_screenshot = screenshot_design(self.design_struct, self.screenshot_dimensions, use_rgb=False)
        # Convert everything except walls(1), sources(3), sinks(4) to air(0)
        self.screenshot = raw_screenshot.copy()
        valid_mask = (self.screenshot == 1) | (self.screenshot == 3) | (self.screenshot == 4)
        self.screenshot[~valid_mask] = 0  # Everything else becomes air
        
        print(f"Screenshot generated: {self.screenshot.shape}")
        print(f"Pixel value distribution: {np.bincount(self.screenshot.flatten())}")
    
    def add_handcrafted_contestant(self, config_path: str):
        """Add handcrafted waypoints from config file."""
        # Make path absolute if it's relative
        if not os.path.isabs(config_path):
            autotweaker_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            config_path = os.path.join(autotweaker_root, config_path)
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        waypoints = config.get('waypoints', [])
        contestant = FirelightContestant("Handcrafted", waypoints, source="handcrafted")
        self.contestants.append(contestant)
        print(f"Added handcrafted contestant with {len(waypoints)} waypoints")
    
    def add_algorithm_contestants(self, algorithm_names: Optional[List[str]] = None, include_all: bool = False):
        """Generate waypoint lists from working algorithms."""
        print("Generating waypoint contestants...")
        
        # Check if "all" was passed as a special flag
        if algorithm_names == ['all']:
            algorithm_names = None
            include_all = True
        
        # Load algorithms from Galapagos tournament results
        working_algorithms = []
        
        # Basic algorithms (always include)
        try:
            basic_tournament = create_default_tournament()
            for generator in basic_tournament.generators:
                if algorithm_names is None or generator.name in algorithm_names or include_all:
                    working_algorithms.append(generator)
        except Exception as e:
            self._log_exception("Loading basic algorithms", e)
        
        # Creative algorithms 
        try:
            creative_tournament = create_quick_creative_tournament()
            if include_all:
                # Include all creative algorithms for comprehensive mode
                for generator in creative_tournament.generators:
                    working_algorithms.append(generator)
            else:
                # Only successful ones for normal mode
                successful_creative = ['QuickFlowField', 'QuickGenetic', 'ImprovedCornerTurning']
                for generator in creative_tournament.generators:
                    if generator.name in successful_creative and (algorithm_names is None or generator.name in algorithm_names):
                        working_algorithms.append(generator)
        except Exception as e:
            self._log_exception("Loading creative algorithms", e)
            
        # Weird algorithms (only if comprehensive mode)
        if include_all:
            try:
                from weird_waypoint_generators import create_weird_tournament
                weird_tournament = create_weird_tournament()
                for generator in weird_tournament.generators:
                    working_algorithms.append(generator)
                print("Added weird algorithms for comprehensive mode")
            except Exception as e:
                self._log_exception("Loading weird algorithms", e)
                
        # Learning algorithms (only if comprehensive mode, if working)
        if include_all:
            try:
                from learning_waypoint_generators import create_learning_tournament
                learning_tournament = create_learning_tournament()
                for generator in learning_tournament.generators:
                    working_algorithms.append(generator)
                print("Added learning algorithms for comprehensive mode")
            except Exception as e:
                self._log_exception("Loading learning algorithms", e)
                
        # Web-inspired algorithms (only if comprehensive mode, if working)  
        if include_all:
            try:
                from web_inspired_generators import create_web_inspired_tournament
                web_tournament = create_web_inspired_tournament()
                for generator in web_tournament.generators:
                    working_algorithms.append(generator)
                print("Added web-inspired algorithms for comprehensive mode")
            except Exception as e:
                self._log_exception("Loading web-inspired algorithms", e)
        
        # Generate waypoints for each algorithm (pure subprocess-based parallelism)
        # Use 1 subprocess = 1 job pattern for true parallelism (no GIL, no threading)
        if working_algorithms:
            print(f"Generating waypoints using subprocess workers (1 process per algorithm)...")
            
            import tempfile
            import json
            import numpy as np
            import subprocess
            import sys
            import os
            
            # Save screenshot to temporary file for subprocess access
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                screenshot_data = {
                    'screenshot': self.screenshot.tolist(),
                    'dimensions': self.screenshot_dimensions
                }
                json.dump(screenshot_data, f)
                screenshot_file = f.name
            
            def run_waypoint_subprocess(generator_name):
                """Run single subprocess for waypoint generation."""
                try:
                    # Find autotweaker root directory
                    current_file = os.path.abspath(__file__)
                    autotweaker_root = os.path.dirname(os.path.dirname(current_file))
                    py_autotweaker_dir = os.path.join(autotweaker_root, 'py_autotweaker')
                    ftlib_test_dir = os.path.join(autotweaker_root, 'ftlib', 'test')
                    
                    cmd = [
                        sys.executable, '-c',
                        f"""
import sys
import os
import json
import numpy as np

# Add paths relative to autotweaker root
autotweaker_root = r'{autotweaker_root}'
sys.path.insert(0, autotweaker_root)
sys.path.insert(0, r'{py_autotweaker_dir}')
sys.path.insert(0, r'{ftlib_test_dir}')

# Change to autotweaker root directory for consistent imports
os.chdir(autotweaker_root)

# Load screenshot data
with open(r'{screenshot_file}', 'r') as f:
    data = json.load(f)
screenshot = np.array(data['screenshot'])
dimensions = tuple(data['dimensions'])

# Import waypoint generation
from py_autotweaker.subprocess_runner import create_generator

# Generate waypoints
generator = create_generator('{generator_name}')
pixel_waypoints = generator.generate_waypoints(screenshot)

# Convert to world coordinates
from py_autotweaker.coordinate_transform import pixel_to_world
world_waypoints = pixel_to_world(pixel_waypoints, dimensions)

# Output results as JSON
import json
result = {{
    'success': True,
    'name': '{generator_name}',
    'waypoints': world_waypoints,
    'count': len(pixel_waypoints)
}}
print(json.dumps(result))
                        """
                    ]
                    
                    # Run subprocess with timeout
                    result = subprocess.run(
                        cmd, 
                        capture_output=True, 
                        text=True, 
                        timeout=60,  # 1 minute timeout per algorithm
                        cwd=os.getcwd()
                    )
                    
                    if result.returncode == 0:
                        # Parse JSON output
                        output_lines = result.stdout.strip().split('\n')
                        json_line = output_lines[-1]  # Last line should be JSON
                        return json.loads(json_line)
                    else:
                        return {
                            'success': False,
                            'name': generator_name,
                            'error': f"Subprocess failed (exit code {result.returncode}): {result.stderr}"
                        }
                        
                except subprocess.TimeoutExpired:
                    return {
                        'success': False,
                        'name': generator_name,
                        'error': f"Waypoint generation timed out after 60 seconds"
                    }
                except Exception as e:
                    return {
                        'success': False,
                        'name': generator_name,
                        'error': f"Unexpected error: {str(e)}"
                    }
            
            try:
                # Run subprocesses in parallel using direct subprocess management
                max_parallel = min(len(working_algorithms), self.max_workers)
                
                results = []
                for i in range(0, len(working_algorithms), max_parallel):
                    batch = working_algorithms[i:i + max_parallel]
                    
                    # Start all processes in batch
                    processes = []
                    for generator in batch:
                        # Build subprocess command
                        autotweaker_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                        py_autotweaker_dir = os.path.join(autotweaker_root, 'py_autotweaker')
                        ftlib_test_dir = os.path.join(autotweaker_root, 'ftlib', 'test')
                        
                        cmd = [
                            sys.executable, '-c',
                            f"""
import sys
import os
import json
import numpy as np

# Add paths relative to autotweaker root
autotweaker_root = r'{autotweaker_root}'
sys.path.insert(0, autotweaker_root)
sys.path.insert(0, r'{py_autotweaker_dir}')
sys.path.insert(0, r'{ftlib_test_dir}')

# Change to autotweaker root directory for consistent imports
os.chdir(autotweaker_root)

# Load screenshot data
with open(r'{screenshot_file}', 'r') as f:
    data = json.load(f)
screenshot = np.array(data['screenshot'])
dimensions = tuple(data['dimensions'])

# Import waypoint generation
from py_autotweaker.subprocess_runner import create_generator

# Generate waypoints
generator = create_generator('{generator.name}')
pixel_waypoints = generator.generate_waypoints(screenshot)

# Convert to world coordinates
from py_autotweaker.coordinate_transform import pixel_to_world
world_waypoints = pixel_to_world(pixel_waypoints, dimensions)

# Output results as JSON
import json
result = {{
    'success': True,
    'name': '{generator.name}',
    'waypoints': world_waypoints,
    'count': len(pixel_waypoints)
}}
print(json.dumps(result))
                            """
                        ]
                        
                        # Start subprocess
                        proc = subprocess.Popen(
                            cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            cwd=autotweaker_root
                        )
                        processes.append((proc, generator.name))
                    
                    # Wait for all processes in batch to complete
                    for proc, generator_name in processes:
                        try:
                            stdout, stderr = proc.communicate(timeout=60)
                            
                            if proc.returncode == 0:
                                # Parse JSON output
                                output_lines = stdout.strip().split('\n')
                                json_line = output_lines[-1]  # Last line should be JSON
                                result = json.loads(json_line)
                                results.append(result)
                            else:
                                results.append({
                                    'success': False,
                                    'name': generator_name,
                                    'error': f"Subprocess failed (exit code {proc.returncode}): {stderr}"
                                })
                        except subprocess.TimeoutExpired:
                            proc.kill()
                            results.append({
                                'success': False,
                                'name': generator_name,
                                'error': f"Waypoint generation timed out after 60 seconds"
                            })
                        except Exception as e:
                            results.append({
                                'success': False,
                                'name': generator_name,
                                'error': f"Unexpected error: {str(e)}"
                            })
                
                # Process all results
                for result in results:
                    if result['success']:
                        contestant = FirelightContestant(result['name'], result['waypoints'], source="algorithm")
                        self.contestants.append(contestant)
                        print(f"  {result['name']}: Generated {result['count']} waypoints")
                    else:
                        print(f"  {result['name']}: Failed - {result['error']}")
            
            finally:
                # Clean up temporary file
                try:
                    os.unlink(screenshot_file)
                except:
                    pass
        
        algorithm_count = len([c for c in self.contestants if c.source == 'algorithm'])
        print(f"Added {algorithm_count} algorithm contestants")
    
    def _start_autotweaker_subprocess(self, contestant: FirelightContestant, run_id: int) -> Dict[str, Any]:
        """Start single autotweaker subprocess with given waypoints."""
        run_start_time = time.time()
        
        try:
            # Create temporary config file - inherit from example config
            base_config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'example', 'job_config.json')
            
            # Load base config to inherit tick parameters  
            if os.path.exists(base_config_path):
                with open(base_config_path, 'r') as f:
                    temp_config = json.load(f)
                # Override waypoints while keeping tick parameters
                temp_config["waypoints"] = contestant.waypoints
            else:
                # Fallback if example config not found
                temp_config = {
                    "waypoints": contestant.waypoints,
                    "tickMinimum": 600,
                    "tickMaximum": 4000
                }
            
            temp_config_path = f"/tmp/firelight_config_{contestant.name}_{run_id}_{os.getpid()}.json"
            with open(temp_config_path, 'w') as f:
                json.dump(temp_config, f, indent=2)
            
            # Build command with absolute paths
            current_file = os.path.abspath(__file__)
            autotweaker_dir = os.path.dirname(os.path.dirname(current_file))
            run_local_script = os.path.join(autotweaker_dir, 'run_local.sh')
            
            cmd = [
                run_local_script,
                'local',
                '-d', str(self.design_id),
                '-c', os.path.abspath(temp_config_path),
                '-a',  # --do-autotweak
                '-t', str(self.timeout_per_run),
                '-n', '1',  # Single thread per run for consistency
                '-w',  # --stop-on-win
                '-k', '0'  # Don't upload during tournament
            ]
            
            # Start subprocess
            process = subprocess.Popen(
                cmd,
                cwd=autotweaker_dir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            return {
                'process': process,
                'temp_config_path': temp_config_path,
                'start_time': run_start_time,
                'timeout': self.timeout_per_run + 30  # Extra buffer for process overhead
            }
            
        except Exception as e:
            # Cleanup on error
            if 'temp_config_path' in locals() and os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
            raise e
    
    def _wait_for_autotweaker_result(self, proc_info: Dict[str, Any]) -> Dict[str, Any]:
        """Wait for autotweaker subprocess to complete and return results."""
        process = proc_info['process']
        temp_config_path = proc_info['temp_config_path']
        run_start_time = proc_info['start_time']
        timeout = proc_info['timeout']
        
        try:
            # Wait for process with timeout
            stdout, stderr = process.communicate(timeout=timeout)
            
            # Cleanup
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
            
            # Parse results from output
            timeout_reached = "time limit reached" in stdout
            
            # Extract final score (simplified parsing)
            final_score = float('inf')
            time_to_solve = None
            
            # Look for score indicators in output
            for line in stdout.split('\n'):
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
                'stdout': stdout,
                'stderr': stderr,
                'return_code': process.returncode
            }
            
        except subprocess.TimeoutExpired:
            # Kill process and cleanup
            process.kill()
            process.communicate()  # Clean up zombie process
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
            # Kill process if still running and cleanup
            try:
                process.kill()
                process.communicate()
            except:
                pass
            if os.path.exists(temp_config_path):
                os.unlink(temp_config_path)
            
            return {
                'time_to_solve': None,
                'timeout_reached': False,
                'final_score': float('inf'),
                'stdout': '',
                'stderr': str(e),
                'return_code': -2
            }

    def _create_config_for_contestant(self, contestant: FirelightContestant) -> Dict[str, Any]:
        """Create autotweaker config for a contestant."""
        return {
            "waypoints": contestant.waypoints,
            "tickMinimum": 600,
            "tickMaximum": 4000
        }

    def _dry_run_validation(self, verbose: bool = True) -> Dict[str, Any]:
        """Perform dry run validation without executing autotweaker runs.
        
        Validates:
        - Design loading and screenshot generation 
        - Contestant waypoint generation
        - Configuration file creation
        - Autotweaker executable availability
        - File system permissions
        
        Returns:
            Validation results with detailed error information
        """
        print("🔍 DRY RUN: Validating Firelight tournament configuration...")
        
        validation_results = {
            'design_valid': True,
            'contestants_valid': True,
            'autotweaker_valid': True,
            'file_permissions_valid': True,
            'imports_valid': True,
            'errors': [],
            'warnings': [],
            'summary': {}
        }
        
        # 0. Validate critical imports first
        try:
            # Test ftlib imports
            import sys
            autotweaker_root = os.path.dirname(os.path.dirname(__file__))
            ftlib_test_dir = os.path.join(autotweaker_root, 'ftlib', 'test')
            
            if ftlib_test_dir not in sys.path:
                sys.path.insert(0, ftlib_test_dir)
            
            from get_design import retrieveDesign, designDomToStruct
            from py_autotweaker.waypoint_generation import create_default_tournament
            from py_autotweaker.screenshot import screenshot_design
            
            print(f"   ✅ Imports: Critical modules accessible")
            validation_results['summary']['imports'] = 'success'
            
        except Exception as e:
            validation_results['imports_valid'] = False
            error_msg = f"Import validation failed: {e}"
            validation_results['errors'].append(error_msg)
            print(f"   ❌ Imports: {error_msg}")
            # Early exit if imports fail since other validations depend on them
            return validation_results
        
        # 1. Validate design and screenshot
        try:
            if self.design_struct is None or self.screenshot is None:
                raise ValueError("Design struct or screenshot not loaded")
            
            validation_results['summary']['design_id'] = self.design_id
            validation_results['summary']['screenshot_shape'] = self.screenshot.shape
            print(f"   ✅ Design: {self.design_id} loaded with screenshot {self.screenshot.shape}")
            
        except Exception as e:
            validation_results['design_valid'] = False
            validation_results['errors'].append(f"Design validation failed: {e}")
            print(f"   ❌ Design: {e}")
        
        # 2. Validate contestants and waypoints
        contestant_status = []
        for contestant in self.contestants:
            try:
                # Validate contestant has waypoints attribute (empty list is valid)
                if not hasattr(contestant, 'waypoints'):
                    raise ValueError(f"Contestant {contestant.name} missing waypoints attribute")
                
                # Validate waypoint format
                for i, wp in enumerate(contestant.waypoints):
                    if not isinstance(wp, dict) or 'x' not in wp or 'y' not in wp:
                        raise ValueError(f"Waypoint {i} invalid format")
                
                # Test config file creation
                config_data = self._create_config_for_contestant(contestant)
                if not config_data or 'waypoints' not in config_data:
                    raise ValueError(f"Config creation failed for {contestant.name}")
                
                contestant_status.append({
                    'name': contestant.name,
                    'waypoint_count': len(contestant.waypoints),
                    'source': contestant.source,
                    'status': 'valid'
                })
                print(f"   ✅ Contestant: {contestant.name} ({len(contestant.waypoints)} waypoints)")
                
            except Exception as e:
                validation_results['contestants_valid'] = False
                error_msg = f"Contestant {contestant.name} failed: {e}"
                validation_results['errors'].append(error_msg)
                contestant_status.append({
                    'name': contestant.name,
                    'status': 'failed',
                    'error': str(e)
                })
                print(f"   ❌ Contestant: {error_msg}")
        
        validation_results['summary']['contestants'] = contestant_status
        
        # 3. Validate autotweaker executable
        try:
            autotweaker_module = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'py_autotweaker')
            if not os.path.exists(autotweaker_module):
                raise FileNotFoundError(f"Autotweaker module not found: {autotweaker_module}")
            
            # Test basic autotweaker availability using module format
            result = subprocess.run([
                sys.executable, '-m', 'py_autotweaker', '--help'
            ], capture_output=True, text=True, timeout=10, cwd=os.path.dirname(os.path.dirname(__file__)))
            
            if result.returncode != 0:
                raise RuntimeError(f"Autotweaker module not functional: {result.stderr}")
            
            print(f"   ✅ Autotweaker: python -m py_autotweaker")
            validation_results['summary']['autotweaker_module'] = autotweaker_module
            
        except Exception as e:
            validation_results['autotweaker_valid'] = False
            error_msg = f"Autotweaker validation failed: {e}"
            validation_results['errors'].append(error_msg)
            print(f"   ❌ Autotweaker: {error_msg}")
        
        # 4. Validate file permissions (for temp files)
        try:
            import tempfile
            with tempfile.NamedTemporaryFile(mode='w+', delete=True, suffix='.json') as f:
                test_config = {'test': 'data', 'waypoints': [{'x': 0, 'y': 0}]}
                json.dump(test_config, f)
                f.flush()
                # Try to read it back
                f.seek(0)
                loaded_data = json.load(f)
                assert loaded_data == test_config
            
            print(f"   ✅ File permissions: Temporary file creation/deletion working")
            
        except Exception as e:
            validation_results['file_permissions_valid'] = False
            error_msg = f"File permissions validation failed: {e}"
            validation_results['errors'].append(error_msg)
            print(f"   ❌ File permissions: {error_msg}")
        
        # Overall validation summary
        all_valid = all([
            validation_results['imports_valid'],
            validation_results['design_valid'],
            validation_results['contestants_valid'],
            validation_results['autotweaker_valid'],
            validation_results['file_permissions_valid']
        ])
        
        print(f"\n🎯 DRY RUN SUMMARY:")
        if all_valid:
            print(f"   ✅ All validations passed - tournament ready to run")
            print(f"   📊 {len(self.contestants)} contestants × {self.runs_per_contestant} runs = {len(self.contestants) * self.runs_per_contestant} total runs")
            estimated_time = (len(self.contestants) * self.runs_per_contestant * self.timeout_per_run) / self.max_workers
            print(f"   ⏱️  Estimated max execution time: {estimated_time:.1f} seconds")
        else:
            error_count = len(validation_results['errors'])
            print(f"   ❌ {error_count} validation errors found - fix before running")
            for error in validation_results['errors']:
                print(f"      • {error}")
        
        return validation_results
    
    def run_tournament(self, verbose: bool = True) -> Dict[str, Any]:
        """Run the complete Firelight tournament."""
        if not self.contestants:
            raise ValueError("No contestants added to tournament")
        
        # Handle dry run mode
        if self.dry_run:
            return self._dry_run_validation(verbose)
        
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
        
        # Execute all runs with subprocess-based parallelism
        # Use 1 subprocess = 1 job pattern for autotweaker runs (true parallelism, no GIL)
        max_parallel = min(len(tasks), self.max_workers)
        
        for i in range(0, len(tasks), max_parallel):
            batch = tasks[i:i + max_parallel]
            
            # Start all processes in batch
            processes = []
            for contestant, run_id in batch:
                proc_info = self._start_autotweaker_subprocess(contestant, run_id)
                processes.append((proc_info, contestant, run_id))
            
            # Wait for all processes in batch to complete
            for proc_info, contestant, run_id in processes:
                try:
                    result = self._wait_for_autotweaker_result(proc_info)
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
                        sys.stdout.flush()
                        
                        # Show subprocess errors for debugging
                        if result['return_code'] != 0:
                            print(f"  ⚠️  Process error (code {result['return_code']}): {result['stderr'][:200]}")
                            if result['stdout']:
                                print(f"  📝 stdout: {result['stdout'][:200]}")
                            sys.stdout.flush()
                
                except Exception as e:
                    self._log_exception(
                        f"Run processing for {contestant.name} Run {run_id+1}",
                        e,
                        f"Completed runs: {completed_runs}/{total_runs}"
                    )
                    completed_runs += 1
        
        # Calculate statistics for all contestants
        for contestant in self.contestants:
            contestant.calculate_statistics()
        
        # Calculate synthetic scores using Galapagos scoring methods
        print("\nCalculating synthetic scores for comparison...")
        for contestant in self.contestants:
            try:
                # Basic Galapagos score (minimal features)
                basic_score = score_waypoint_list(self.screenshot, contestant.waypoints, 
                                                penalize_skippable=True, feature_flags=None)
                
                # Improved Galapagos score (all features)
                improved_score = improved_score_waypoint_list(self.screenshot, contestant.waypoints)
                
                # Store in contestant for later use
                contestant.synthetic_scores = {
                    'basic_score': basic_score,
                    'improved_score': improved_score
                }
                
                print(f"  {contestant.name}: Basic={basic_score:.1f}, Improved={improved_score:.1f}")
                
            except Exception as e:
                print(f"  {contestant.name}: Synthetic scoring failed - {e}")
                contestant.synthetic_scores = {
                    'basic_score': float('inf'),
                    'improved_score': float('inf')
                }
        
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
        
        # Log exception summary
        if self.exception_count > 0:
            print(f"\n⚠️  Tournament completed with {self.exception_count} exception(s)")
            print(f"    Full exception log: {self.exception_log_file}")
        
        return results
    
    def print_results(self, results: Dict[str, Any]):
        """Print formatted tournament results with dual rankings and TSV output."""
        print("\n" + "=" * 80)
        print("FIRELIGHT TOURNAMENT RESULTS")
        print("=" * 80)
        
        info = results['tournament_info']
        print(f"Design ID: {info['design_id']}")
        print(f"Total execution time: {info['total_time']:.1f}s")
        print(f"Total runs: {info['total_runs']} ({info['runs_per_contestant']} per contestant)")
        print()
        
        contestants = results['contestants']
        
        # Generate TSV output
        print("TSV DATA (copy-paste friendly):")
        print("-" * 80)
        
        # TSV Headers
        tsv_headers = [
            "name", "source", "waypoints", "success_rate", "successes", "total_runs",
            "avg_solve_time", "median_solve_time", "min_solve_time", "max_solve_time", "solve_time_stdev",
            "best_final_score", "avg_final_score", "timeouts", "failures",
            "basic_synthetic_score", "improved_synthetic_score"
        ]
        print("\t".join(tsv_headers))
        
        # TSV Data
        for contestant in contestants:
            stats = contestant.statistics
            synthetic = getattr(contestant, 'synthetic_scores', {'basic_score': 'N/A', 'improved_score': 'N/A'})
            
            tsv_values = [
                str(contestant.name),
                str(contestant.source),
                str(len(contestant.waypoints)),
                f"{stats['success_rate']:.4f}",
                str(stats['successes']),
                str(stats['total_runs']),
                f"{stats['avg_solve_time']:.2f}" if stats['avg_solve_time'] is not None else "N/A",
                f"{stats['median_solve_time']:.2f}" if stats['median_solve_time'] is not None else "N/A",
                f"{stats['min_solve_time']:.2f}" if stats['min_solve_time'] is not None else "N/A",
                f"{stats['max_solve_time']:.2f}" if stats['max_solve_time'] is not None else "N/A",
                f"{stats['solve_time_stdev']:.2f}" if stats['solve_time_stdev'] is not None else "N/A",
                f"{stats['best_final_score']:.2f}" if stats['best_final_score'] != float('inf') else "inf",
                f"{stats['avg_final_score']:.2f}" if stats['avg_final_score'] != float('inf') else "inf",
                str(stats['timeouts']),
                str(stats['failures']),
                f"{synthetic['basic_score']:.2f}" if isinstance(synthetic['basic_score'], (int, float)) else "N/A",
                f"{synthetic['improved_score']:.2f}" if isinstance(synthetic['improved_score'], (int, float)) else "N/A"
            ]
            print("\t".join(tsv_values))
        
        print("-" * 80)
        print()
        
        # Ranking 1: By Real Performance (success rate, then average solve time)
        print("RANKING 1: BY REAL PERFORMANCE")
        print("-" * 50)
        real_ranking = sorted(contestants, 
                             key=lambda c: (-c.statistics['success_rate'], 
                                          c.statistics['avg_solve_time'] or float('inf')))
        
        print(f"{'Rank':<4} {'Name':<20} {'Success%':<8} {'Avg Time':<10} {'Best Score':<10}")
        print("-" * 50)
        for i, contestant in enumerate(real_ranking, 1):
            stats = contestant.statistics
            success_pct = f"{stats['success_rate']*100:.1f}%"
            avg_time = f"{stats['avg_solve_time']:.1f}s" if stats['avg_solve_time'] is not None else "N/A"
            best_score = f"{stats['best_final_score']:.1f}" if stats['best_final_score'] != float('inf') else "∞"
            print(f"{i:<4} {contestant.name:<20} {success_pct:<8} {avg_time:<10} {best_score:<10}")
        
        print()
        
        # Ranking 2: By Basic Synthetic Score (lower is better)
        print("RANKING 2: BY BASIC SYNTHETIC SCORE (Galapagos basic)")
        print("-" * 50)
        basic_ranking = sorted([c for c in contestants if hasattr(c, 'synthetic_scores')], 
                              key=lambda c: c.synthetic_scores['basic_score'])
        
        print(f"{'Rank':<4} {'Name':<20} {'Basic Score':<12} {'Success%':<8} {'Avg Time':<10}")
        print("-" * 50)
        for i, contestant in enumerate(basic_ranking, 1):
            stats = contestant.statistics
            basic_score = f"{contestant.synthetic_scores['basic_score']:.1f}"
            success_pct = f"{stats['success_rate']*100:.1f}%"
            avg_time = f"{stats['avg_solve_time']:.1f}s" if stats['avg_solve_time'] is not None else "N/A"
            print(f"{i:<4} {contestant.name:<20} {basic_score:<12} {success_pct:<8} {avg_time:<10}")
        
        print()
        
        # Ranking 3: By Improved Synthetic Score (lower is better)
        print("RANKING 3: BY IMPROVED SYNTHETIC SCORE (Galapagos improved)")
        print("-" * 50)
        improved_ranking = sorted([c for c in contestants if hasattr(c, 'synthetic_scores')], 
                                 key=lambda c: c.synthetic_scores['improved_score'])
        
        print(f"{'Rank':<4} {'Name':<20} {'Improved Score':<14} {'Success%':<8} {'Avg Time':<10}")
        print("-" * 50)
        for i, contestant in enumerate(improved_ranking, 1):
            stats = contestant.statistics
            improved_score = f"{contestant.synthetic_scores['improved_score']:.1f}"
            success_pct = f"{stats['success_rate']*100:.1f}%"
            avg_time = f"{stats['avg_solve_time']:.1f}s" if stats['avg_solve_time'] is not None else "N/A"
            print(f"{i:<4} {contestant.name:<20} {improved_score:<14} {success_pct:<8} {avg_time:<10}")
        
        print("=" * 80)
        
        # Correlation analysis
        successful_contestants = [c for c in contestants if c.statistics['successes'] > 0 and hasattr(c, 'synthetic_scores')]
        if len(successful_contestants) >= 2:
            print("\nCORRELATION ANALYSIS:")
            print("-" * 30)
            
            # Extract data for correlation
            real_scores = []  # Lower avg solve time = better performance
            basic_scores = []
            improved_scores = []
            
            for contestant in successful_contestants:
                if contestant.statistics['avg_solve_time'] is not None:
                    real_scores.append(contestant.statistics['avg_solve_time'])
                    basic_scores.append(contestant.synthetic_scores['basic_score'])
                    improved_scores.append(contestant.synthetic_scores['improved_score'])
            
            if len(real_scores) >= 2:
                import statistics
                
                # Calculate correlation (basic approximation)
                def simple_correlation(x, y):
                    if len(x) != len(y) or len(x) < 2:
                        return 0
                    x_mean = statistics.mean(x)
                    y_mean = statistics.mean(y)
                    
                    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
                    x_var = sum((xi - x_mean) ** 2 for xi in x)
                    y_var = sum((yi - y_mean) ** 2 for yi in y)
                    
                    if x_var == 0 or y_var == 0:
                        return 0
                    return numerator / (x_var * y_var) ** 0.5
                
                basic_corr = simple_correlation(real_scores, basic_scores)
                improved_corr = simple_correlation(real_scores, improved_scores)
                
                print(f"Real vs Basic Synthetic:    {basic_corr:.3f}")
                print(f"Real vs Improved Synthetic: {improved_corr:.3f}")
                print("(Positive correlation = synthetic predicts reality well)")
            else:
                print("Not enough data for correlation analysis")
        else:
            print("\nNot enough successful contestants for correlation analysis")


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
    parser.add_argument('--comprehensive', action='store_true', help='Full comprehensive mode: all algorithms, max workers, long timeouts')
    
    args = parser.parse_args()
    
    # Adjust settings for comprehensive mode (full everything)
    if args.comprehensive:
        args.runs = max(args.runs, 15)  # More runs for statistical significance
        args.timeout = max(args.timeout, 600)  # 10 minutes per run
        import multiprocessing
        args.workers = multiprocessing.cpu_count()  # Use ALL available cores
        args.algorithms = None  # Use all algorithms
        print("🔬 COMPREHENSIVE MODE: Maximum settings for complete evaluation")
    elif args.quick:
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
    tournament.add_algorithm_contestants(args.algorithms, include_all=args.comprehensive if hasattr(args, 'comprehensive') else False)
    
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