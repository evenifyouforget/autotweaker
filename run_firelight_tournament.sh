#!/bin/bash

# Firelight Tournament Runner
# End-to-end waypoint testing using actual autotweaker pipeline

set -e

# Setup paths
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/py_autotweaker:$(pwd)/ftlib/test"

# Default parameters
DESIGN_ID="12710291"
RUNS_PER_CONTESTANT="3"
TIMEOUT_PER_RUN="120"  # 2 minutes default (user requested 300s for full run)
MAX_WORKERS="2"
HANDCRAFTED_CONFIG="example/job_config.json"

# Parse command line arguments
QUICK_MODE=false
COMPREHENSIVE_MODE=false
ALGORITHMS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--design-id)
            DESIGN_ID="$2"
            shift 2
            ;;
        -r|--runs)
            RUNS_PER_CONTESTANT="$2" 
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT_PER_RUN="$2"
            shift 2
            ;;
        -w|--workers)
            MAX_WORKERS="$2"
            shift 2
            ;;
        -c|--handcrafted-config)
            HANDCRAFTED_CONFIG="$2"
            shift 2
            ;;
        -a|--algorithms)
            ALGORITHMS="$2"
            shift 2
            ;;
        -q|--quick)
            QUICK_MODE=true
            shift
            ;;
        -f|--full)
            # Full production settings
            RUNS_PER_CONTESTANT="10"
            TIMEOUT_PER_RUN="300"
            MAX_WORKERS="4"
            shift
            ;;
        -C|--comprehensive)
            # Full comprehensive everything mode
            COMPREHENSIVE_MODE=true
            RUNS_PER_CONTESTANT="15"
            TIMEOUT_PER_RUN="600"  # 10 minutes per run
            MAX_WORKERS="6"
            ALGORITHMS="all"  # Special flag for all algorithms
            shift
            ;;
        -h|--help)
            echo "Firelight Tournament: End-to-End Waypoint Pipeline Testing"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  -d, --design-id ID        Design ID to test (default: $DESIGN_ID)"
            echo "  -r, --runs N              Runs per contestant (default: $RUNS_PER_CONTESTANT)"  
            echo "  -t, --timeout SECONDS     Timeout per run (default: $TIMEOUT_PER_RUN)"
            echo "  -w, --workers N           Max parallel workers (default: $MAX_WORKERS)"
            echo "  -c, --handcrafted-config  Path to handcrafted config (default: $HANDCRAFTED_CONFIG)"
            echo "  -a, --algorithms LIST     Specific algorithms to test (space-separated)"
            echo "  -q, --quick               Quick test mode (2 runs, 60s timeout)"
            echo "  -f, --full                Full production mode (10 runs, 300s timeout, 4 workers)"
            echo "  -C, --comprehensive       COMPREHENSIVE mode (15 runs, 600s timeout, all algorithms)"
            echo "  -h, --help               Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 -q                         # Quick test"
            echo "  $0 -f                         # Full production run"
            echo "  $0 -C                         # Full comprehensive everything"
            echo "  $0 -d 12345678                # Test different design"
            echo "  $0 -a \"Null CornerTurning\"   # Test specific algorithms"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Apply mode settings
if [ "$QUICK_MODE" = true ]; then
    RUNS_PER_CONTESTANT="2"
    TIMEOUT_PER_RUN="60"
    echo "🚀 Quick mode enabled: $RUNS_PER_CONTESTANT runs, ${TIMEOUT_PER_RUN}s timeout"
elif [ "$COMPREHENSIVE_MODE" = true ]; then
    echo "🔬 COMPREHENSIVE mode enabled: $RUNS_PER_CONTESTANT runs, ${TIMEOUT_PER_RUN}s timeout, all algorithms"
fi

echo "=================================="
echo "FIRELIGHT TOURNAMENT"  
echo "=================================="
echo "Design ID: $DESIGN_ID"
echo "Runs per contestant: $RUNS_PER_CONTESTANT"
echo "Timeout per run: ${TIMEOUT_PER_RUN}s"
echo "Max workers: $MAX_WORKERS"
echo "Handcrafted config: $HANDCRAFTED_CONFIG"
if [ -n "$ALGORITHMS" ]; then
    echo "Specific algorithms: $ALGORITHMS"
fi
echo ""

# Check prerequisites
echo "🔍 Checking prerequisites..."

if [ ! -f "$HANDCRAFTED_CONFIG" ]; then
    echo "❌ Handcrafted config not found: $HANDCRAFTED_CONFIG"
    exit 1
fi

if [ ! -d "ftlib/test" ]; then
    echo "❌ ftlib test directory not found. Initialize git submodule:"
    echo "   git submodule update --init --recursive"
    exit 1
fi

# Test imports
python3 -c "
import sys
sys.path.append('.')
sys.path.append('py_autotweaker') 
sys.path.append('ftlib/test')

try:
    from get_design import retrieveDesign, designDomToStruct
    from py_autotweaker.waypoint_generation import create_default_tournament
    from py_autotweaker.screenshot import screenshot_design
    print('✅ All imports successful')
except ImportError as e:
    print(f'❌ Import error: {e}')
    exit(1)
" || exit 1

echo "✅ Prerequisites check passed"
echo ""

# Build command
CMD="python3 -c \"
import sys
import os
sys.path.append('.')
sys.path.append('py_autotweaker')
sys.path.append('ftlib/test')

from py_autotweaker.firelight_tournament import create_firelight_tournament
import json

# Create tournament
tournament = create_firelight_tournament(
    design_id=$DESIGN_ID,
    runs_per_contestant=$RUNS_PER_CONTESTANT,
    timeout_per_run=$TIMEOUT_PER_RUN,
    max_workers=$MAX_WORKERS
)

# Add handcrafted contestant
tournament.add_handcrafted_contestant('$HANDCRAFTED_CONFIG')

# Add algorithm contestants"

if [ -n "$ALGORITHMS" ]; then
    CMD="$CMD
algorithms = '$ALGORITHMS'.split()
tournament.add_algorithm_contestants(algorithms)"
else
    CMD="$CMD
tournament.add_algorithm_contestants()"
fi

CMD="$CMD

# Run tournament
if not tournament.contestants:
    print('❌ No contestants added to tournament!')
    exit(1)

results = tournament.run_tournament()
tournament.print_results(results)

# Save results
import time
from pathlib import Path

results_dir = Path('firelight_results')
results_dir.mkdir(exist_ok=True)

timestamp = time.strftime('%Y%m%d_%H%M%S')
results_file = results_dir / f'firelight_{$DESIGN_ID}_{timestamp}.json'

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

print(f'\\n💾 Results saved to: {results_file}')
\""

echo "🏁 Starting Firelight tournament..."
echo ""

# Execute the tournament
eval $CMD