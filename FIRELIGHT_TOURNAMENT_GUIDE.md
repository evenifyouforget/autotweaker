# Firelight Tournament System

## Overview

The Firelight Tournament system provides end-to-end validation of waypoint generation algorithms by testing them through the complete autotweaker pipeline. Unlike the Galapagos tournaments (which score waypoints using synthetic metrics), Firelight tournaments measure real-world performance by running the actual autotweaker optimization process.

## Tournament Types

### Galapagos Tournaments
- **Purpose**: Test waypoint quality using synthetic scoring functions
- **Speed**: Fast (seconds to minutes)
- **Coverage**: Can test many algorithms quickly
- **Files**: `experimental_comprehensive_tournament.py`, `multithreaded_tournament.py`
- **Use case**: Algorithm development and rapid iteration

### Firelight Tournaments  
- **Purpose**: End-to-end validation through actual autotweaker pipeline
- **Speed**: Slow (minutes to hours depending on runs/timeouts)
- **Coverage**: Real-world performance measurement
- **Files**: `firelight_tournament.py`, `run_firelight_tournament.sh`
- **Use case**: Final validation of promising algorithms

## System Architecture

### Core Components

1. **FirelightContestant**: Represents a waypoint algorithm or handcrafted configuration
2. **FirelightTournament**: Manages the complete tournament lifecycle
3. **Screenshot Normalization**: Converts RGB screenshots to algorithm-compatible format
4. **Subprocess Management**: Handles autotweaker execution with timeouts
5. **Statistical Analysis**: Analyzes results across multiple runs

### Workflow

1. **Setup Phase**:
   - Load design and generate screenshot
   - Normalize screenshot colors (RGB → algorithm format)
   - Generate waypoints from algorithms
   - Add handcrafted baseline from config file

2. **Execution Phase**:
   - Create temporary config files for each contestant
   - Run autotweaker subprocess with timeouts
   - Parse output for scores and solve detection
   - Handle timeouts and errors gracefully

3. **Analysis Phase**:
   - Calculate statistics across multiple runs
   - Handle timeout bounds in statistical analysis
   - Rank contestants by success rate and solve time
   - Generate comprehensive results

## Usage

### Quick Test
```bash
./run_firelight_tournament.sh --quick
```

### Full Production Run
```bash
./run_firelight_tournament.sh --full
```

### Custom Configuration
```bash
./run_firelight_tournament.sh \
  --design-id 12710291 \
  --runs 10 \
  --timeout 300 \
  --workers 4 \
  --algorithms "Null CornerTurning QuickGenetic"
```

### Python API
```python
from py_autotweaker.firelight_tournament import create_firelight_tournament

tournament = create_firelight_tournament(
    design_id=12710291,
    runs_per_contestant=5,
    timeout_per_run=180,
    max_workers=2
)

tournament.add_handcrafted_contestant('example/job_config.json')
tournament.add_algorithm_contestants(['Null', 'CornerTurning'])

results = tournament.run_tournament()
tournament.print_results(results)
```

## Key Features

### Statistical Analysis
- Success rate (percentage of runs that solve)
- Average solve time with standard deviation
- Best score achieved across all runs
- Proper handling of timeout bounds (censored observations)

### Robust Execution
- Subprocess-based parallel execution (no Python GIL limitations)
- Configurable workers with auto-detection of CPU cores
- Per-run timeouts to prevent hangs
- Process isolation to prevent crashes
- Automatic cleanup of temporary files and zombie processes

### Comprehensive Logging
- Real-time progress reporting
- Detailed error handling and logging
- Result persistence to JSON files
- Export-friendly format for further analysis

## Important Considerations

### Coordinate Systems
- **World Coordinates**: Used by autotweaker (e.g., -600, +600 range)
- **Screenshot Coordinates**: Used by algorithms (e.g., 0-400 pixel range)
- **Critical**: Algorithms generate pixel coordinates but autotweaker expects world coordinates
- **Solution**: Coordinate transformation needed between systems

### Performance Expectations
- Example job (design 12710291): 2-5 minutes typical completion
- Difficult designs: May timeout even with good waypoints
- Multiple runs needed: Randomness requires statistical sampling
- Resource usage: Each worker runs full autotweaker instance

### Current Status

#### Working Components ✅
- Complete Firelight tournament framework
- Subprocess-based parallel execution with process isolation
- Screenshot generation and color normalization
- Statistical analysis and reporting
- **Coordinate transformation**: Bidirectional pixel ↔ world coordinate mapping
- Handcrafted waypoint integration
- Strongly-typed data structures with validation
- Static analysis and automated quality checking

#### Recent Major Improvements ✅
- **Threading → Subprocess**: Converted from ThreadPoolExecutor to subprocess-based parallelism
- **True CPU Parallelism**: Bypassed Python GIL limitations for better performance
- **Coordinate System Fixed**: Implemented precise coordinate transformation using screenshot.py formulas
- **Tournament Consolidation**: Unified duplicate tournament implementations
- **Enhanced Typing**: Added comprehensive type annotations with validation
- **Path Resolution**: Fixed directory-independent execution

#### Testing Results
- **Null vs Handcrafted**: Both achieve similar high scores (1e+300) on design 12710291
- **Subprocess working**: autotweaker execution successful with proper argument format
- **Performance bug fixed**: Thread count conversion issue resolved

## Future Enhancements

### Enhanced Analysis
- Performance profiling of individual algorithms
- Level difficulty classification
- Algorithm recommendation based on level characteristics
- Multi-design tournament support

### Optimization
- Cached screenshot generation
- Parallel algorithm waypoint generation  
- Smart timeout adjustment based on design complexity
- Resource usage monitoring

## Integration with Galapagos

The two tournament systems complement each other:

1. **Development Cycle**: Use Galapagos for rapid algorithm development
2. **Validation Cycle**: Use Firelight for final validation of promising candidates
3. **Research Pipeline**: Galapagos → algorithmic improvements → Firelight validation
4. **Production Selection**: Firelight results determine real-world algorithm deployment

This dual approach ensures both development velocity and real-world validation.