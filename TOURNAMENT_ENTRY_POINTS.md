# Waypoint Generation Tournament Entry Points

## 🚀 Quick Start

The main entry point for running waypoint generation tournaments is:

```bash
./run_waypoint_tournament.sh
```

## 📋 Available Modes

### 1. **Synthetic Mode** (Default, Fast)
Test algorithms on hand-crafted synthetic levels:
```bash
./run_waypoint_tournament.sh synthetic
./run_waypoint_tournament.sh                    # same as above
```

### 2. **Real Level Mode** (Requires Internet)
Test algorithms on actual Fantastic Contraption levels:
```bash
./run_waypoint_tournament.sh real --max-levels 5
```

### 3. **Mixed Mode** (Comprehensive)
Test on both synthetic and real levels:
```bash
./run_waypoint_tournament.sh mixed --max-levels 3
```

### 4. **List Algorithms**
Show available algorithms:
```bash
./run_waypoint_tournament.sh list
./run_waypoint_tournament.sh list --advanced    # include advanced algorithms
```

## ⚙️ Options

| Option | Description | Example |
|--------|-------------|---------|
| `--max-levels N` | Maximum real levels to test | `--max-levels 10` |
| `--advanced` | Include advanced algorithms (slower) | `--advanced` |
| `--quiet` | Reduce output verbosity | `--quiet` |
| `--help` | Show help message | `--help` |

## 🧠 Available Algorithms

### Basic Algorithms (Fast)
- **Null**: Empty waypoint list (baseline)
- **CornerTurning**: Recursive corner detection algorithm

### Advanced Algorithms (Slower, use `--advanced`)
- **MedialAxis**: Uses skeleton of passable areas
- **Voronoi**: Places waypoints at maximal distance from walls  
- **OptimizedSearch**: Simulated annealing optimization

## 📊 Example Usage

### Quick Test (< 5 seconds)
```bash
./run_waypoint_tournament.sh synthetic
```

### Comprehensive Test (< 30 seconds)
```bash
./run_waypoint_tournament.sh mixed --max-levels 5
```

### Advanced Algorithm Comparison (< 2 minutes)
```bash
./run_waypoint_tournament.sh synthetic --advanced
```

### Real Level Deep Dive (< 1 minute)
```bash
./run_waypoint_tournament.sh real --max-levels 10
```

## 🔧 Alternative Entry Points

### Direct Python Usage
```bash
# Via module
python3 -m py_autotweaker.tournament_runner --mode synthetic

# Direct script
python3 py_autotweaker/tournament_runner.py --mode real --max-levels 5
```

### Individual Test Scripts
```bash
# Basic synthetic test
python3 test_waypoints.py

# Real level test
python3 test_real_levels.py

# Advanced algorithms
python3 test_quick_advanced.py
```

## 🏆 Understanding Results

The tournament ranks algorithms by **total score** (lower is better):

- **Score 0.00**: Perfect performance (no waypoints needed or optimal waypoints)
- **Score > 0**: Higher scores indicate suboptimal waypoint placement
- **Skippable**: Invalid waypoints that can be bypassed (heavily penalized)
- **Errors**: Algorithm failures or crashes

### Key Metrics
- **Total Score**: Sum across all test cases (primary ranking)
- **Avg Score**: Average score per test case
- **Avg Time**: Algorithm execution time
- **Avg WP**: Average waypoints generated per level

## 🐛 Troubleshooting

### Common Issues

**"ftlib submodule not initialized"**
```bash
git submodule update --init --recursive
```

**"get_design module not found"**
- Ensure ftlib submodule is initialized
- Check internet connection for real level testing

**"Missing Python dependencies"**
```bash
pip install numpy scipy scikit-image networkx matplotlib
```

**Long execution times**
- Use `--max-levels 3` to limit real level testing
- Avoid `--advanced` flag for quick tests
- Use `synthetic` mode for fastest testing

### Debug Mode
For detailed error information:
```bash
python3 py_autotweaker/tournament_runner.py --mode synthetic
```

## 📁 File Structure

```
autotweaker/
├── run_waypoint_tournament.sh          # Main entry point (shell script)
├── py_autotweaker/
│   ├── tournament_runner.py            # Main tournament runner (Python)
│   ├── waypoint_generation.py          # Basic algorithms
│   ├── advanced_waypoint_generators.py # Advanced algorithms
│   ├── waypoint_scoring.py             # Scoring system
│   └── waypoint_test_runner.py         # Test infrastructure
├── test_waypoints.py                   # Basic test script
├── test_real_levels.py                 # Real level test script
└── maze_like_levels.tsv                # Real level IDs database
```

The shell script (`run_waypoint_tournament.sh`) provides the most user-friendly interface with automatic environment setup, dependency checking, and colored output.