#!/usr/bin/env python3
"""
Simple test script for the waypoint generation system.

This script tests the waypoint generation algorithms on synthetic test cases
and validates that the system works correctly.
"""

import sys
import os

# Add py_autotweaker to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'py_autotweaker'))

from waypoint_test_runner import test_synthetic_levels, run_quick_test

def main():
    print("="*60)
    print("WAYPOINT GENERATION SYSTEM TEST")
    print("="*60)
    
    print("\n1. Testing synthetic levels...")
    try:
        synthetic_results = test_synthetic_levels(verbose=True)
        
        if "tournament_results" in synthetic_results:
            print(f"\n✓ Synthetic test completed successfully!")
            print(f"  - Tested {synthetic_results['test_case_count']} synthetic levels")
            
            # Show which generators worked
            results = synthetic_results["tournament_results"]
            working_generators = [name for name, data in results.items() 
                                if data['error_count'] == 0]
            print(f"  - {len(working_generators)} generators working: {', '.join(working_generators)}")
        else:
            print("✗ Synthetic test failed")
            
    except Exception as e:
        print(f"✗ Synthetic test error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("TEST COMPLETED")
    print("="*60)
    
    # Optionally test real levels (commented out since it requires level loading)
    print("\nNote: Real level testing requires the ftlib backend to be built.")
    print("To test real levels, uncomment the code below and ensure ftlib is available.")
    
    """
    print("\n2. Testing real levels...")
    try:
        real_results = run_quick_test(max_levels=2, verbose=True)
        if "error" not in real_results:
            print(f"✓ Real level test completed!")
        else:
            print(f"✗ Real level test failed: {real_results['error']}")
    except Exception as e:
        print(f"✗ Real level test error: {e}")
    """

if __name__ == "__main__":
    main()