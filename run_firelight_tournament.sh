#!/bin/bash

# Firelight Tournament Runner
# End-to-end waypoint testing using actual autotweaker pipeline

set -e

# Setup paths and ftlib environment
cd ftlib
source environment.sh
cd ..
export PYTHONPATH="${PYTHONPATH}:$(pwd):$(pwd)/py_autotweaker:$(pwd)/ftlib/test"

# Default parameters
DESIGN_ID="12710291"
RUNS_PER_CONTESTANT="3"
TIMEOUT_PER_RUN="120"  # 2 minutes default (user requested 300s for full run)
MAX_WORKERS=""  # Auto-detect by default
HANDCRAFTED_CONFIG="example/job_config.json"

# Parse command line arguments
QUICK_MODE=false
COMPREHENSIVE_MODE=false
ALGORITHMS=""
USER_TIMEOUT=""
USER_RUNS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--design-id)
            DESIGN_ID="$2"
            shift 2
            ;;
        -r|--runs)
            RUNS_PER_CONTESTANT="$2"
            USER_RUNS="$2"
            shift 2
            ;;
        -t|--timeout)
            TIMEOUT_PER_RUN="$2"
            USER_TIMEOUT="$2"
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
        -f|--full|-C|--comprehensive)
            # FULL MODE: Maximum comprehensive everything 
            COMPREHENSIVE_MODE=true
            echo "🔬 FULL MODE: Maximum comprehensive settings enabled"
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
            echo "  -f, --full                FULL MODE: Maximum comprehensive (15 runs, 600s timeout, all algorithms, max workers)"
            echo "  -C, --comprehensive       Same as --full (alias for compatibility)"
            echo "  -h, --help               Show this help"
            echo ""
            echo "Examples:"
            echo "  $0 -q                         # Quick test"
            echo "  $0 -f                         # FULL MODE: maximum comprehensive"
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

# Apply mode settings (but preserve user-specified values)
if [ "$QUICK_MODE" = true ]; then
    if [ -z "$USER_RUNS" ]; then
        RUNS_PER_CONTESTANT="2"
    fi
    if [ -z "$USER_TIMEOUT" ]; then
        TIMEOUT_PER_RUN="60"
    fi
    echo "🚀 Quick mode enabled: $RUNS_PER_CONTESTANT runs, ${TIMEOUT_PER_RUN}s timeout"
elif [ "$COMPREHENSIVE_MODE" = true ]; then
    if [ -z "$USER_RUNS" ]; then
        RUNS_PER_CONTESTANT="15"
    fi
    if [ -z "$USER_TIMEOUT" ]; then
        TIMEOUT_PER_RUN="600"  # 10 minutes per run
    fi
    if [ -z "$MAX_WORKERS" ]; then
        MAX_WORKERS=""  # Auto-detect all available cores
    fi
    if [ -z "$ALGORITHMS" ]; then
        ALGORITHMS="all"  # Special flag for all algorithms
    fi
    echo "🔬 FULL mode enabled: $RUNS_PER_CONTESTANT runs, ${TIMEOUT_PER_RUN}s timeout, all algorithms"
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
python3.13 -c "
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

# Use simple Python runner to avoid bash string issues
# Auto-detect max workers if not specified
if [ -z "$MAX_WORKERS" ]; then
    ACTUAL_MAX_WORKERS=$(python3.13 -c "import multiprocessing; print(multiprocessing.cpu_count())")
else
    ACTUAL_MAX_WORKERS="$MAX_WORKERS"
fi

CMD="python3.13 py_autotweaker/run_firelight_simple.py $DESIGN_ID $RUNS_PER_CONTESTANT $TIMEOUT_PER_RUN $ACTUAL_MAX_WORKERS $HANDCRAFTED_CONFIG"

# Add algorithms parameter  
if [ -n "$ALGORITHMS" ]; then
    CMD="$CMD $ALGORITHMS"
else
    CMD="$CMD none"
fi

echo "🏁 Starting Firelight tournament..."
echo ""

# Execute the tournament
eval $CMD