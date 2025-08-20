#!/bin/bash

# Complete Tournament Runner (Everything)
# Runs dry runs first to catch errors early, then executes full tournaments
# Uses existing run_tournament.sh and run_firelight_tournament.sh scripts

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/tournament_logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Default parameters  
RUNS_PER_CONTESTANT=5
TIMEOUT_PER_RUN=180
MAX_WORKERS=$(nproc)  # Use all available cores for full mode
DESIGN_ID=12710291

# Create log directory
mkdir -p "$LOG_DIR"

# Log files
MAIN_LOG="${LOG_DIR}/comprehensive_tournament_${TIMESTAMP}.log"
GALAPAGOS_LOG="${LOG_DIR}/galapagos_${TIMESTAMP}.log"
FIRELIGHT_LOG="${LOG_DIR}/firelight_${TIMESTAMP}.log"

echo "🚀 COMPLETE TOURNAMENT RUNNER (EVERYTHING)" | tee "$MAIN_LOG"
echo "=================================" | tee -a "$MAIN_LOG"
echo "Timestamp: $(date)" | tee -a "$MAIN_LOG"
echo "Log directory: $LOG_DIR" | tee -a "$MAIN_LOG"
echo "Parameters:" | tee -a "$MAIN_LOG"
echo "  Design ID: $DESIGN_ID" | tee -a "$MAIN_LOG"
echo "  Runs per contestant: $RUNS_PER_CONTESTANT" | tee -a "$MAIN_LOG"
echo "  Timeout per run: $TIMEOUT_PER_RUN" | tee -a "$MAIN_LOG"
echo "  Max workers: $MAX_WORKERS" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Change to script directory
cd "$SCRIPT_DIR"

# Check if ftlib submodule is initialized
if [[ ! -f "ftlib/test/get_design.py" ]]; then
    echo "⚠️ WARNING: ftlib submodule not initialized. Attempting to initialize..." | tee -a "$MAIN_LOG"
    if ! git submodule update --init --recursive; then
        echo "❌ FATAL: Failed to initialize ftlib submodule" | tee -a "$MAIN_LOG"
        echo "Please run: git submodule update --init --recursive" | tee -a "$MAIN_LOG"
        exit 1
    fi
    echo "✅ ftlib submodule initialized successfully" | tee -a "$MAIN_LOG"
fi

# Setup Python path for ftlib imports
export PYTHONPATH="$(pwd)/py_autotweaker:$(pwd)/ftlib/test"

# Function to run command with logging and error handling
run_with_logging() {
    local cmd="$1"
    local log_file="$2"
    local description="$3"
    
    echo "⏳ $description..." | tee -a "$MAIN_LOG"
    echo "Command: $cmd" | tee -a "$MAIN_LOG"
    echo "Log file: $log_file" | tee -a "$MAIN_LOG"
    echo ""
    
    if eval "$cmd" 2>&1 | tee "$log_file"; then
        echo "✅ $description completed successfully" | tee -a "$MAIN_LOG"
        echo "" | tee -a "$MAIN_LOG"
        return 0
    else
        echo "❌ $description failed - check $log_file for details" | tee -a "$MAIN_LOG"
        echo "" | tee -a "$MAIN_LOG"
        return 1
    fi
}

# Phase 1: Galapagos Tournament Dry Run (Fast synthetic test)
echo "=== PHASE 1: GALAPAGOS DRY RUN ===" | tee -a "$MAIN_LOG"
GALAPAGOS_DRY_CMD="./run_tournament.sh synthetic --fast --quiet"

if ! run_with_logging "$GALAPAGOS_DRY_CMD" "$GALAPAGOS_LOG" "Galapagos tournament dry run"; then
    echo "🛑 STOPPING: Galapagos dry run failed" | tee -a "$MAIN_LOG"
    exit 1
fi

# Phase 2: Firelight Tournament Dry Run
echo "=== PHASE 2: FIRELIGHT DRY RUN ===" | tee -a "$MAIN_LOG"
FIRELIGHT_DRY_CMD="./run_firelight_tournament.sh --quick --design-id $DESIGN_ID"

if ! run_with_logging "$FIRELIGHT_DRY_CMD" "$FIRELIGHT_LOG" "Firelight tournament dry run"; then
    echo "🛑 STOPPING: Firelight dry run failed" | tee -a "$MAIN_LOG"
    exit 1
fi

echo "🎯 DRY RUNS COMPLETED SUCCESSFULLY!" | tee -a "$MAIN_LOG"
echo "All validations passed - proceeding with full tournaments" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

# Phase 3: Full Galapagos Tournament (All levels, all algorithms)
echo "=== PHASE 3: FULL GALAPAGOS TOURNAMENT ===" | tee -a "$MAIN_LOG"
GALAPAGOS_FULL_CMD="./run_tournament.sh --full"

if ! run_with_logging "$GALAPAGOS_FULL_CMD" "$GALAPAGOS_LOG" "Full Galapagos tournament"; then
    echo "⚠️ WARNING: Galapagos tournament failed, but continuing with Firelight" | tee -a "$MAIN_LOG"
fi

# Phase 4: Full Firelight Tournament (All algorithms + handcrafted, generous timeouts)
echo "=== PHASE 4: FULL FIRELIGHT TOURNAMENT ===" | tee -a "$MAIN_LOG"
FIRELIGHT_FULL_CMD="./run_firelight_tournament.sh --design-id $DESIGN_ID --full"

if ! run_with_logging "$FIRELIGHT_FULL_CMD" "$FIRELIGHT_LOG" "Full Firelight tournament"; then
    echo "⚠️ WARNING: Firelight tournament failed" | tee -a "$MAIN_LOG"
fi

# Final summary
echo "=== COMPREHENSIVE TOURNAMENT SUMMARY ===" | tee -a "$MAIN_LOG"
echo "Execution completed at: $(date)" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"
echo "📁 Log files:" | tee -a "$MAIN_LOG"
echo "  Main log: $MAIN_LOG" | tee -a "$MAIN_LOG"
echo "  Galapagos log: $GALAPAGOS_LOG" | tee -a "$MAIN_LOG"
echo "  Firelight log: $FIRELIGHT_LOG" | tee -a "$MAIN_LOG"
echo "" | tee -a "$MAIN_LOG"

if [[ -f "$GALAPAGOS_LOG" ]] && [[ -f "$FIRELIGHT_LOG" ]]; then
    echo "✅ Both tournament logs generated successfully" | tee -a "$MAIN_LOG"
    echo "📊 Review the individual log files for detailed results" | tee -a "$MAIN_LOG"
else
    echo "⚠️ Some log files may be missing - check for errors above" | tee -a "$MAIN_LOG"
fi

echo "" | tee -a "$MAIN_LOG"
echo "🎉 COMPLETE TOURNAMENT RUNNER (EVERYTHING) COMPLETED!" | tee -a "$MAIN_LOG"