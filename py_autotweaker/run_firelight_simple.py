#!/usr/bin/env python3.13
"""
Simple Firelight tournament runner to avoid bash string issues.
"""

import sys
import os
import json
import time
from pathlib import Path

# Add paths
sys.path.append('.')
sys.path.append('py_autotweaker')
sys.path.append('ftlib/test')

from py_autotweaker.firelight_tournament import create_firelight_tournament

def main():
    # Force unbuffered output for real-time progress
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
    
    if len(sys.argv) < 7:
        print("Usage: run_firelight_simple.py <design_id> <runs> <timeout> <workers> <config> <algorithms>")
        sys.exit(1)
    
    design_id = int(sys.argv[1])
    runs_per_contestant = int(sys.argv[2])
    timeout_per_run = int(sys.argv[3])
    max_workers = int(sys.argv[4])
    handcrafted_config = sys.argv[5]
    algorithms = sys.argv[6] if sys.argv[6] != "none" else None
    
    print(f"🔥 Starting Firelight tournament for design {design_id}")
    print(f"   Runs per contestant: {runs_per_contestant}")
    print(f"   Timeout per run: {timeout_per_run}s")
    print(f"   Max workers: {max_workers}")
    print(f"   Algorithms: {algorithms or 'default'}")
    sys.stdout.flush()
    
    # Create tournament
    print("📊 Creating tournament...")
    sys.stdout.flush()
    tournament = create_firelight_tournament(
        design_id=design_id,
        runs_per_contestant=runs_per_contestant,
        timeout_per_run=timeout_per_run,
        max_workers=max_workers
    )
    
    # Add handcrafted contestant
    print("🎯 Adding handcrafted contestant...")
    sys.stdout.flush()
    tournament.add_handcrafted_contestant(handcrafted_config)
    
    # Add algorithm contestants
    print("🤖 Adding algorithm contestants...")
    sys.stdout.flush()
    if algorithms == "all":
        tournament.add_algorithm_contestants(['all'])
    elif algorithms:
        algorithm_list = algorithms.split(',')
        tournament.add_algorithm_contestants(algorithm_list)
    else:
        tournament.add_algorithm_contestants()
    
    # Run tournament
    if not tournament.contestants:
        print('❌ No contestants added to tournament!')
        sys.exit(1)
    
    print(f"🏃 Running tournament with {len(tournament.contestants)} contestants...")
    sys.stdout.flush()
    results = tournament.run_tournament()
    
    print("📈 Displaying results...")
    sys.stdout.flush()
    tournament.print_results(results)
    
    # Save results
    results_dir = Path('firelight_results')
    results_dir.mkdir(exist_ok=True)
    
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    results_file = results_dir / f'firelight_{design_id}_{timestamp}.json'
    
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

if __name__ == "__main__":
    main()