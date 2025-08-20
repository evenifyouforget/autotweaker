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
    
    # Create generator using reflection
    generator = create_generator(generator_name)
    
    # Execute the algorithm
    start_time = time.time()
    waypoints = generator.generate_waypoints(screenshot)
    execution_time = time.time() - start_time
    
    # Score the waypoints
    from improved_waypoint_scoring import improved_score_waypoint_list
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