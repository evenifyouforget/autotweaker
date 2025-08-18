#!/usr/bin/env bash
#
# Comprehensive Waypoint Tournament Runner
#
# This script runs the FULL waypoint tournament with ALL algorithms on ALL 100 levels
# from the maze_like_levels.tsv database with comprehensive analysis.
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

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Parse arguments
FAST_MODE=false
MAX_LEVELS=100

while [[ $# -gt 0 ]]; do
    case $1 in
        --fast)
            FAST_MODE=true
            MAX_LEVELS=20
            shift
            ;;
        --max-levels)
            MAX_LEVELS="$2"
            shift 2
            ;;
        --help|-h)
            cat << EOF
Comprehensive Waypoint Tournament Runner

USAGE:
    $0 [OPTIONS]

OPTIONS:
    --fast          Fast mode: 20 levels instead of full 100 (for testing)
    --max-levels N  Custom number of levels to test
    --help          Show this help

DESCRIPTION:
    This runs the FULL comprehensive tournament with:
    - ALL creative algorithms (8+ algorithms total)
    - Full database testing (up to 100 levels)
    - Comprehensive statistical analysis
    - Results saving with detailed breakdowns
    - Performance benchmarking and rankings

ALGORITHMS INCLUDED:
    Basic:
    - Null (baseline empty waypoints)
    - CornerTurning (recursive corner detection)
    
    Creative:
    - Genetic (evolutionary optimization)
    - FlowField (potential field analysis)  
    - SwarmIntelligence (particle swarm optimization)
    - AdaptiveRandom (learning-based random sampling)
    - ImprovedCornerTurning (physics-based balloon expansion)
    - MedialAxis (skeleton-based placement)
    - Voronoi (distance transform optimization)
    - OptimizedSearch (simulated annealing)

EXAMPLES:
    $0                    # Full tournament: ALL 8+ algorithms on ALL 100 levels
    $0 --fast            # Fast test: 8+ algorithms on 20 levels
    $0 --max-levels 50   # Custom: 8+ algorithms on 50 levels

EOF
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Header
echo "========================================================================"
echo "           COMPREHENSIVE WAYPOINT GENERATION TOURNAMENT"
echo "========================================================================"
echo ""

# Configuration summary
if [[ "$FAST_MODE" == "true" ]]; then
    print_status "FAST MODE: Testing $MAX_LEVELS levels with all creative algorithms"
else
    print_status "FULL MODE: Testing $MAX_LEVELS levels with all creative algorithms"
fi

echo ""
print_status "Tournament Configuration:"
echo "  - Algorithms: ALL creative algorithms (8+ total)"
echo "  - Test Levels: $MAX_LEVELS real Fantastic Contraption levels"
echo "  - Scoring: Improved evidence-based scoring with ant simulation"
echo "  - Analysis: Comprehensive statistical analysis with results saving"
echo "  - Expected Runtime: $(($MAX_LEVELS / 5)) - $(($MAX_LEVELS / 2)) minutes"

echo ""

# Check if we're in the right directory
if [[ ! -f "maze_like_levels.tsv" ]]; then
    print_error "Must run from autotweaker directory (maze_like_levels.tsv not found)"
    exit 1
fi

# Check if ftlib submodule is initialized
if [[ ! -f "ftlib/test/get_design.py" ]]; then
    print_status "Initializing ftlib submodule..."
    if ! git submodule update --init --recursive; then
        print_error "Failed to initialize ftlib submodule"
        exit 1
    fi
    print_success "ftlib submodule initialized"
fi

# Set up ftlib environment
print_status "Setting up ftlib environment..."
if [[ -f "ftlib/environment.sh" ]]; then
    cd ftlib
    source environment.sh
    cd ..
    print_success "ftlib environment loaded"
fi

# Run the comprehensive tournament
print_status "Starting comprehensive tournament..."
echo ""

CMD="python3 py_autotweaker/full_tournament_runner.py"

if [[ "$FAST_MODE" == "true" ]]; then
    CMD="$CMD --fast"
else
    CMD="$CMD --max-levels $MAX_LEVELS --creative"
fi

print_status "Executing: $CMD"
echo ""

if eval $CMD; then
    echo ""
    print_success "Comprehensive tournament completed successfully!"
    
    # Show results location
    RESULTS_DIR="py_autotweaker/../results"
    if [[ -d "$RESULTS_DIR" ]]; then
        LATEST_RESULT=$(ls -t $RESULTS_DIR/tournament_results_*.json | head -n1 2>/dev/null || echo "")
        if [[ -n "$LATEST_RESULT" ]]; then
            print_success "Detailed results saved to: $LATEST_RESULT"
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
echo "COMPREHENSIVE TOURNAMENT COMPLETED"
echo ""
echo "What was tested:"
echo "  ✓ $MAX_LEVELS real Fantastic Contraption levels from maze_like_levels.tsv"
echo "  ✓ 8+ waypoint generation algorithms (basic + creative)"
echo "  ✓ Evidence-based scoring with ant movement simulation"  
echo "  ✓ Statistical analysis and performance ranking"
echo "  ✓ Results saved to JSON for further analysis"
echo ""
echo "Check the results above for:"
echo "  - Algorithm performance rankings"
echo "  - Success rates and reliability metrics"
echo "  - Level difficulty categorization"
echo "  - Execution time analysis"
echo "========================================================================"