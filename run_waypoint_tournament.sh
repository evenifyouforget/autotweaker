#!/usr/bin/env bash
#
# Waypoint Generation Tournament Runner
#
# This script runs waypoint generation algorithm tournaments on Fantastic Contraption levels.
# It automatically handles ftlib setup and provides various testing modes.
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
Waypoint Generation Tournament Runner

USAGE:
    $0 [MODE] [OPTIONS]

MODES:
    synthetic           Run on synthetic test levels (fast, default)
    real               Run on real FC levels (requires internet)
    mixed              Run on both synthetic and real levels
    list               List available algorithms

OPTIONS:
    --max-levels N     Maximum real levels to test (default: 10)
    --advanced         Include advanced algorithms (slower)
    --quiet            Reduce output verbosity
    --help             Show this help message

EXAMPLES:
    $0                          # Quick synthetic test
    $0 synthetic --advanced     # Synthetic with advanced algorithms
    $0 real --max-levels 5      # Test 5 real levels
    $0 mixed --max-levels 3     # Mixed test with 3 real levels
    $0 list --advanced          # List all algorithms

REQUIREMENTS:
    - Internet connection (for real level mode)
    - ftlib submodule initialized
    - Python packages: numpy, scipy, scikit-image, networkx

EOF
}

# Parse command line arguments
MODE="synthetic"
MAX_LEVELS=10
ADVANCED_FLAG=""
QUIET_FLAG=""

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
echo "                 WAYPOINT GENERATION TOURNAMENT"
echo "========================================================================"
echo ""

# Check if we're in the right directory
if [[ ! -f "maze_like_levels.tsv" ]]; then
    print_error "Must run from autotweaker directory (maze_like_levels.tsv not found)"
    exit 1
fi

# Check if ftlib submodule is initialized
if [[ ! -f "ftlib/test/get_design.py" ]]; then
    print_warning "ftlib submodule not initialized. Attempting to initialize..."
    if ! git submodule update --init --recursive; then
        print_error "Failed to initialize ftlib submodule"
        exit 1
    fi
    print_success "ftlib submodule initialized"
fi

# Set up ftlib environment if needed (for real level testing)
if [[ "$MODE" == "real" || "$MODE" == "mixed" ]]; then
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

# Build command
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
else
    echo ""
    print_error "Tournament failed!"
    exit 1
fi

# Footer
echo ""
echo "========================================================================"
echo "Tips:"
echo "  - Use 'synthetic' mode for quick algorithm development"
echo "  - Use 'real' mode to test on actual FC levels" 
echo "  - Use 'mixed' mode for comprehensive evaluation"
echo "  - Add --advanced for more sophisticated algorithms"
echo "  - See detailed logs above for algorithm performance"
echo "========================================================================"