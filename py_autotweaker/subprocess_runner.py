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

# Add current directory to path for imports
sys.path.append(os.path.dirname(__file__))

def load_task(task_file: str):
    """Load task data from pickle file."""
    with open(task_file, 'rb') as f:
        return pickle.load(f)

def create_generator(generator_name: str):
    """Create a generator instance by name."""
    # Import generators based on name
    try:
        if generator_name == "Null":
            from waypoint_generation import NullGenerator
            return NullGenerator()
        elif generator_name == "CornerTurning":
            from waypoint_generation import CornerTurningGenerator
            return CornerTurningGenerator()
        elif generator_name.startswith("Quick"):
            from quick_creative_generators import (
                QuickGeneticGenerator, QuickFlowFieldGenerator, 
                QuickSwarmGenerator, QuickAdaptiveGenerator
            )
            if generator_name == "QuickGenetic":
                return QuickGeneticGenerator()
            elif generator_name == "QuickFlowField":
                return QuickFlowFieldGenerator()
            elif generator_name == "QuickSwarm":
                return QuickSwarmGenerator()
            elif generator_name == "QuickAdaptive":
                return QuickAdaptiveGenerator()
            else:
                raise ValueError(f"Unknown quick generator: {generator_name}")
        elif generator_name == "ImprovedCornerTurning":
            from improved_corner_turning import ImprovedCornerTurningGenerator
            return ImprovedCornerTurningGenerator()
        else:
            # Try to load from creative generators
            try:
                from creative_waypoint_generators import (
                    GeneticWaypointGenerator, FlowFieldGenerator,
                    SwarmIntelligenceGenerator, AdaptiveRandomGenerator
                )
                if generator_name == "Genetic":
                    return GeneticWaypointGenerator()
                elif generator_name == "FlowField":
                    return FlowFieldGenerator()
                elif generator_name == "SwarmIntelligence":
                    return SwarmIntelligenceGenerator()
                elif generator_name == "AdaptiveRandom":
                    return AdaptiveRandomGenerator()
                else:
                    # Try weird generators
                    from weird_waypoint_generators import (
                        ChaosWaypointGenerator, AntiWaypointGenerator, MegaWaypointGenerator,
                        FibonacciWaypointGenerator, MirrorWaypointGenerator, PrimeNumberWaypointGenerator,
                        TimeBasedWaypointGenerator, CornerMagnifierGenerator, EdgeHuggerGenerator
                    )
                    if generator_name == "Chaos":
                        return ChaosWaypointGenerator()
                    elif generator_name == "Anti":
                        return AntiWaypointGenerator()
                    elif generator_name == "Mega":
                        return MegaWaypointGenerator()
                    elif generator_name == "Fibonacci":
                        return FibonacciWaypointGenerator()
                    elif generator_name == "Mirror":
                        return MirrorWaypointGenerator()
                    elif generator_name == "Prime":
                        return PrimeNumberWaypointGenerator()
                    elif generator_name == "TimeBased":
                        return TimeBasedWaypointGenerator()
                    elif generator_name == "CornerMagnifier":
                        return CornerMagnifierGenerator()
                    elif generator_name == "EdgeHugger":
                        return EdgeHuggerGenerator()
                    else:
                        raise ValueError(f"Unknown generator: {generator_name}")
            except ImportError:
                raise ValueError(f"Unknown generator: {generator_name}")
    
    except ImportError as e:
        raise ImportError(f"Failed to import generator {generator_name}: {e}")


def execute_algorithm_task(generator_name: str, test_case_name: str, screenshot):
    """Execute a single algorithm on a test case."""
    
    # Create generator using shared function
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
        'score': score,
        'waypoints': waypoints,
        'execution_time': execution_time
    }

def main():
    """Main subprocess entry point."""
    
    if len(sys.argv) != 2:
        print(json.dumps({
            'error': 'Usage: subprocess_runner.py <task_file>'
        }))
        sys.exit(1)
    
    task_file = sys.argv[1]
    
    try:
        # Load task
        task_data = load_task(task_file)
        generator_name = task_data['generator_name']
        test_case_name = task_data['test_case_name']
        screenshot = pickle.loads(task_data['screenshot_data'])
        
        # Execute task
        result = execute_algorithm_task(generator_name, test_case_name, screenshot)
        
        # Output result as JSON
        print(json.dumps(result, default=str))
        sys.exit(0)
        
    except Exception as e:
        # Output error as JSON
        error_result = {
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()