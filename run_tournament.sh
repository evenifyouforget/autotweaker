#!/usr/bin/env bash
#
# Unified Waypoint Generation Tournament Runner
#
# This is the main entry point for all waypoint generation tournaments.
# It supports synthetic testing, real level testing, and comprehensive analysis.
#

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Unified Waypoint Generation Tournament Runner

USAGE:
    $0 [MODE] [OPTIONS]

MODES:
    synthetic           Run on synthetic test levels (fast, default)
    real               Run on real FC levels (requires internet)
    mixed              Run on both synthetic and real levels  
    list               List available algorithms

OPTIONS:
    --max-levels N     Maximum real levels to test (default: 10)
    --advanced         Include all creative algorithms (8+ total, slower)
    --full             Test all 100 REAL levels with creative algorithms
    --comprehensive    Use comprehensive analysis system (saves detailed results)
    --multithreaded    Use multithreaded execution with timeouts (faster, recommended)
    --fast             Quick test mode (20 levels max, fast algorithms)
    --timeout N        Timeout per algorithm in seconds (default: 10)
    --quiet            Reduce output verbosity
    --help             Show this help message

EXAMPLES:
    $0                          # Quick synthetic test (2 basic algorithms)
    $0 synthetic --advanced     # Synthetic with all 8+ creative algorithms  
    $0 real --max-levels 5      # Test 5 real levels (basic algorithms)
    $0 real --advanced          # Test 10 real levels with creative algorithms
    $0 --full                   # FULL TEST: All 100 real levels with all algorithms
    $0 --multithreaded --advanced  # FAST: Multithreaded with all algorithms (recommended)
    $0 --multithreaded --fast   # Quick multithreaded demo (20 levels, 10s timeout)
    $0 list --advanced          # List all algorithms including creative ones

SPECIAL MODES:
    --full              Equivalent to: real --max-levels 100 --advanced --comprehensive
    --comprehensive     Uses advanced analysis system with JSON result saving
    --fast              Limits to 20 levels and uses optimized quick algorithms

REQUIREMENTS:
    - Internet connection (for real level mode)
    - ftlib submodule initialized  
    - Python packages: numpy, scipy, scikit-image, networkx

ALGORITHM TYPES:
    Basic (always included):
    - Null: Empty waypoint list (baseline)
    - CornerTurning: Recursive corner detection

    Creative (--advanced flag):
    - Genetic: Evolutionary optimization  
    - FlowField: Potential field analysis
    - SwarmIntelligence: Particle swarm optimization
    - AdaptiveRandom: Learning-based random sampling
    - ImprovedCornerTurning: Physics-based balloon expansion
    - And more (8+ algorithms total)

EOF
}

# Parse command line arguments
MODE="synthetic"
MAX_LEVELS=10
ADVANCED_FLAG=""
COMPREHENSIVE_FLAG=""
MULTITHREADED_FLAG=""
QUIET_FLAG=""
FAST_MODE=false
TIMEOUT=10

while [[ $# -gt 0 ]]; do
    case $1 in
        synthetic|real|mixed|list)
            MODE="$1"
            shift
            ;;
        --max-levels)
            MAX_LEVELS="$2"
            shift 2
            ;;
        --advanced)
            ADVANCED_FLAG="--advanced"
            shift
            ;;
        --full)
            MODE="real"
            MAX_LEVELS=100
            ADVANCED_FLAG="--advanced"
            COMPREHENSIVE_FLAG="--comprehensive"
            print_status "FULL MODE: Testing all 100 real levels with comprehensive analysis"
            shift
            ;;
        --comprehensive)
            COMPREHENSIVE_FLAG="--comprehensive"
            shift
            ;;
        --multithreaded)
            MULTITHREADED_FLAG="--multithreaded"
            shift
            ;;
        --timeout)
            TIMEOUT="$2"
            shift 2
            ;;
        --fast)
            FAST_MODE=true
            MAX_LEVELS=20
            ADVANCED_FLAG="--advanced"  # Include advanced algorithms in fast mode
            shift
            ;;
        --quiet)
            QUIET_FLAG="--quiet"
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Header
echo "========================================================================"
echo "           UNIFIED WAYPOINT GENERATION TOURNAMENT"
echo "========================================================================"
echo ""

# Configuration summary
if [[ "$FAST_MODE" == "true" ]]; then
    print_status "FAST MODE: Testing up to $MAX_LEVELS levels with optimized algorithms"
elif [[ "$COMPREHENSIVE_FLAG" == "--comprehensive" ]]; then
    print_status "COMPREHENSIVE MODE: Advanced analysis with detailed result saving"
fi

if [[ "$MODE" == "real" || "$MODE" == "mixed" ]]; then
    print_status "Will test real Fantastic Contraption levels from maze_like_levels.tsv"
fi

if [[ -n "$ADVANCED_FLAG" ]]; then
    print_status "Including creative algorithms (8+ total)"
fi

echo ""

# Check if we're in the right directory
if [[ ! -f "maze_like_levels.tsv" ]]; then
    print_error "Must run from autotweaker directory (maze_like_levels.tsv not found)"
    exit 1
fi

# Check if ftlib submodule is initialized for real testing
if [[ "$MODE" == "real" || "$MODE" == "mixed" ]]; then
    if [[ ! -f "ftlib/test/get_design.py" ]]; then
        print_warning "ftlib submodule not initialized. Attempting to initialize..."
        if ! git submodule update --init --recursive; then
            print_error "Failed to initialize ftlib submodule"
            exit 1
        fi
        print_success "ftlib submodule initialized"
    fi
    
    # Set up ftlib environment
    print_status "Setting up ftlib environment for real level testing..."
    if [[ -f "ftlib/environment.sh" ]]; then
        cd ftlib
        source environment.sh
        cd ..
        print_success "ftlib environment loaded"
    else
        print_warning "ftlib/environment.sh not found, continuing without ftlib environment"
    fi
fi

# Check Python dependencies
print_status "Checking Python dependencies..."
MISSING_DEPS=""

for dep in numpy scipy networkx; do
    if ! python3 -c "import $dep" 2>/dev/null; then
        MISSING_DEPS="$MISSING_DEPS $dep"
    fi
done

if [[ -n "$MISSING_DEPS" ]]; then
    print_warning "Missing Python dependencies:$MISSING_DEPS"
    print_status "You may need to install them with: pip install$MISSING_DEPS"
fi

# Build command based on mode
if [[ "$MULTITHREADED_FLAG" == "--multithreaded" ]]; then
    # Use multithreaded tournament system
    CMD="python3 py_autotweaker/comprehensive_multithreaded_tournament.py"
    
    if [[ "$MODE" == "real" || "$MODE" == "mixed" ]]; then
        CMD="$CMD --real --max-levels $MAX_LEVELS"
    else
        CMD="$CMD --synthetic"
    fi
    
    if [[ -n "$ADVANCED_FLAG" ]]; then
        # Advanced includes creative algorithms, also add weird ones
        CMD="$CMD"  # Creative algorithms included by default
    else
        CMD="$CMD --no-creative --no-weird"  # Only basic algorithms
    fi
    
    CMD="$CMD --timeout $TIMEOUT"
    
    if [[ -n "$QUIET_FLAG" ]]; then
        CMD="$CMD > /dev/null"
    fi
    
elif [[ "$COMPREHENSIVE_FLAG" == "--comprehensive" ]]; then
    # Use comprehensive tournament system
    CMD="python3 py_autotweaker/full_tournament_runner.py"
    
    if [[ "$FAST_MODE" == "true" ]]; then
        CMD="$CMD --fast"
    else
        CMD="$CMD --max-levels $MAX_LEVELS --creative"
    fi
    
    if [[ -n "$QUIET_FLAG" ]]; then
        CMD="$CMD --quiet"
    fi
    
    print_status "Using comprehensive tournament system"
else
    # Use regular tournament system
    CMD="python3 py_autotweaker/tournament_runner.py --mode $MODE"
    
    if [[ -n "$MAX_LEVELS" && ("$MODE" == "real" || "$MODE" == "mixed") ]]; then
        CMD="$CMD --max-levels $MAX_LEVELS"
    fi
    
    if [[ -n "$ADVANCED_FLAG" ]]; then
        CMD="$CMD $ADVANCED_FLAG"
    fi
    
    if [[ -n "$QUIET_FLAG" ]]; then
        CMD="$CMD $QUIET_FLAG"
    fi
    
    print_status "Using standard tournament system"
fi

# Show what we're about to run
print_status "Running tournament in '$MODE' mode..."
if [[ "$MODE" == "real" || "$MODE" == "mixed" ]]; then
    print_status "Will test up to $MAX_LEVELS real levels"
fi
if [[ -n "$ADVANCED_FLAG" ]]; then
    print_status "Including advanced algorithms (this may take longer)"
fi

echo ""

# Run the tournament
print_status "Executing: $CMD"
echo ""

if eval $CMD; then
    echo ""
    print_success "Tournament completed successfully!"
    
    # Show results location if comprehensive mode
    if [[ "$COMPREHENSIVE_FLAG" == "--comprehensive" ]]; then
        RESULTS_DIR="results"
        if [[ -d "$RESULTS_DIR" ]]; then
            LATEST_RESULT=$(ls -t $RESULTS_DIR/tournament_results_*.json | head -n1 2>/dev/null || echo "")
            if [[ -n "$LATEST_RESULT" ]]; then
                print_success "Detailed results saved to: $LATEST_RESULT"
            fi
        fi
    fi
else
    echo ""
    print_error "Tournament failed!"
    exit 1
fi

# Footer
echo ""
echo "========================================================================"
echo "TOURNAMENT COMPLETED"
echo ""
echo "Usage Tips:"
echo "  - Use 'synthetic' mode for quick algorithm development and testing"
echo "  - Use 'real' mode to test on actual Fantastic Contraption levels" 
echo "  - Use 'mixed' mode for comprehensive evaluation on both types"
echo "  - Add --advanced to include creative algorithms (8+ total)"
echo "  - Use --full for complete testing: all 100 levels with all algorithms"
echo "  - Use --comprehensive for detailed analysis and JSON result saving"
echo "  - Use --fast for quick demonstrations (20 levels, optimized algorithms)"
echo ""
echo "Entry Points:"
echo "  ./run_tournament.sh --multithreaded --advanced  # FASTEST: Parallel execution (recommended)"
echo "  ./run_tournament.sh --full                      # Complete 100-level tournament"  
echo "  ./run_tournament.sh --multithreaded --fast      # Quick parallel demo"
echo "========================================================================"