#!/usr/bin/env python3

"""
Simple test script for Firelight tournament.
"""

import sys
import os

# Setup paths
sys.path.append('.')
sys.path.append('py_autotweaker')
sys.path.append('ftlib/test')

from py_autotweaker.firelight_tournament import create_firelight_tournament
import json
import time
from pathlib import Path

def main():
    print("🔥 FIRELIGHT TOURNAMENT TEST")
    print("=" * 50)
    
    # Create tournament with minimal settings for testing
    tournament = create_firelight_tournament(
        design_id=12710291,
        runs_per_contestant=2,  # Minimal for testing
        timeout_per_run=60,     # Short timeout for testing 
        max_workers=2
    )
    
    # Add handcrafted contestant
    print("📋 Adding handcrafted contestant...")
    tournament.add_handcrafted_contestant('example/job_config.json')
    
    # Add specific algorithms for testing
    print("🤖 Adding algorithm contestants...")
    algorithms = ['Null', 'CornerTurning']  # Only working algorithms
    tournament.add_algorithm_contestants(algorithms)
    
    if not tournament.contestants:
        print('❌ No contestants added to tournament!')
        return 1
    
    print(f"✅ Tournament ready with {len(tournament.contestants)} contestants")
    for contestant in tournament.contestants:
        print(f"   - {contestant.name} ({contestant.source}): {len(contestant.waypoints)} waypoints")
    
    # Run tournament
    print("\n🏁 Starting tournament...")
    results = tournament.run_tournament()
    tournament.print_results(results)
    
    # Save results
    results_dir = Path('firelight_results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'firelight_test_{timestamp}.json'
    
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
    
    print(f'\n💾 Results saved to: {results_file}')
    return 0

if __name__ == "__main__":
    try:
        exit(main())
    except KeyboardInterrupt:
        print("\n⚠️ Tournament interrupted by user")
        exit(1)
    except Exception as e:
        print(f"\n❌ Tournament failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)