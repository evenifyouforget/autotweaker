#!/usr/bin/env python3.13
"""
Subprocess runner for isolated algorithm execution.

This script is called by the multithreaded tournament to execute individual
algorithm tasks in isolation, preventing hangs or crashes from affecting
the main tournament process.
"""

import sys
import os
import json
import pickle
import time
import traceback
import importlib

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

def load_task(task_file: str):
    """Load task data from pickle file."""
    with open(task_file, 'rb') as f:
        return pickle.load(f)

def create_generator(generator_name: str):
    """Create a generator instance by name using shared reflection."""
    from waypoint_generation import create_generator as shared_create_generator
    return shared_create_generator(generator_name)

def execute_algorithm_task(generator_name: str, test_case_name: str, screenshot):
    """Execute a single algorithm on a test case."""
    
    # Fix import path for subprocess execution
    import sys
    import os
    
    # Add py_autotweaker to Python path to resolve relative imports
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if current_dir not in sys.path:
        sys.path.insert(0, current_dir)
    
    # Add parent directory for py_autotweaker package imports  
    parent_dir = os.path.dirname(current_dir)
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
        
    # Create generator using reflection
    generator = create_generator(generator_name)
    
    # Execute the algorithm
    start_time = time.time()
    waypoints = generator.generate_waypoints(screenshot)
    execution_time = time.time() - start_time
    
    # Score the waypoints (use absolute import)
    try:
        from improved_waypoint_scoring import improved_score_waypoint_list
    except ImportError:
        from py_autotweaker.improved_waypoint_scoring import improved_score_waypoint_list
    score = improved_score_waypoint_list(screenshot, waypoints, penalize_skippable=True)
    
    return {
        'generator_name': generator_name,
        'test_case_name': test_case_name,
        'waypoints': waypoints,
        'score': score,
        'execution_time': execution_time,
        'success': True
    }

def main():
    """Main entry point for subprocess execution."""
    if len(sys.argv) != 4:
        print("Usage: subprocess_runner.py <task_file> <result_file> <timeout>")
        sys.exit(1)
    
    task_file = sys.argv[1]
    result_file = sys.argv[2]
    timeout = float(sys.argv[3])
    
    try:
        # Load task
        task = load_task(task_file)
        generator_name = task['generator_name']
        test_case_name = task['test_case_name']
        screenshot = task['screenshot']
        
        # Execute with timeout handling
        start_time = time.time()
        result = execute_algorithm_task(generator_name, test_case_name, screenshot)
        
        # Save result
        with open(result_file, 'wb') as f:
            pickle.dump(result, f)
            
    except Exception as e:
        # Log full exception details to help debugging
        import traceback
        import datetime
        
        # Create error log directory
        import os
        os.makedirs('tournament_errors', exist_ok=True)
        
        # Log to file with timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        error_log_path = f'tournament_errors/subprocess_error_{timestamp}_{generator_name}_{test_case_name}.log'
        
        with open(error_log_path, 'w') as log_file:
            log_file.write(f"Subprocess Error: {generator_name} on {test_case_name}\n")
            log_file.write(f"Timestamp: {timestamp}\n")
            log_file.write(f"Exception Type: {type(e).__name__}\n")
            log_file.write(f"Exception Message: {str(e)}\n")
            log_file.write(f"Full Traceback:\n{traceback.format_exc()}\n")
            log_file.write(f"Task Data: {task}\n")
        
        # Also print first few errors to console
        error_count_file = 'tournament_errors/error_count.txt'
        if os.path.exists(error_count_file):
            with open(error_count_file, 'r') as f:
                count = int(f.read().strip())
        else:
            count = 0
        
        count += 1
        with open(error_count_file, 'w') as f:
            f.write(str(count))
            
        if count <= 3:  # Print first 3 errors to console
            print(f"\n❌ SUBPROCESS ERROR #{count}: {generator_name} on {test_case_name}")
            print(f"   Exception: {type(e).__name__}: {str(e)}")
            print(f"   Full error log: {error_log_path}")
        
        # Categorize the failure
        error_type = "unknown_error"
        if "Unknown generator" in str(e):
            error_type = "import_error"
        elif "timeout" in str(e).lower():
            error_type = "timeout"  
        elif "generate_waypoints" in str(e):
            error_type = "generation_error"
        elif "ImportError" in str(type(e).__name__):
            error_type = "import_error"
        elif "ValueError" in str(type(e).__name__):
            error_type = "validation_error"
        elif "TimeoutError" in str(type(e).__name__):
            error_type = "timeout"
        elif "NotImplementedError" in str(type(e).__name__):
            error_type = "abstract_class"
            
        # Save error result
        error_result = {
            'generator_name': task.get('generator_name', 'Unknown') if 'task' in locals() else 'Unknown',
            'test_case_name': task.get('test_case_name', 'Unknown') if 'task' in locals() else 'Unknown',
            'waypoints': [],
            'score': float('inf'),
            'execution_time': 0.0,
            'success': False,
            'error': str(e),
            'error_type': error_type,
            'traceback': traceback.format_exc()
        }
        
        with open(result_file, 'wb') as f:
            pickle.dump(error_result, f)

if __name__ == "__main__":
    main()