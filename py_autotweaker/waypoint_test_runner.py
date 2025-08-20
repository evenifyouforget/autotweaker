"""
Test runner for waypoint generation algorithms using real level data.

This module loads level data, generates screenshots, and runs waypoint generation
tournaments to evaluate different algorithms.
"""

import sys
import os
import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import json

# Add the ftlib test directory to the path to import get_design
ftlib_test_path = os.path.join(os.path.dirname(__file__), '..', 'ftlib', 'test')
sys.path.append(ftlib_test_path)

try:
    from get_design import FCDesignStruct, retrieveLevel, designDomToStruct
    
    # Create a compatibility wrapper function
    def get_design_by_id(level_id):
        level_dom = retrieveLevel(level_id, is_design=False)
        if level_dom is None:
            return None
        return designDomToStruct(level_dom)
    
except ImportError:
    print("Warning: get_design module not found. Some functionality may be limited.")
    FCDesignStruct = None
    get_design_by_id = None

try:
    # Try relative imports first (when run as module)
    from .screenshot import screenshot_design, WORLD_MIN_X, WORLD_MAX_X, WORLD_MIN_Y, WORLD_MAX_Y
    from .waypoint_generation import WaypointTournament, create_default_tournament
    from .waypoint_scoring import score_waypoint_list, check_waypoint_non_skippability
except ImportError:
    # Fall back to absolute imports (when run as script)
    from screenshot import screenshot_design, WORLD_MIN_X, WORLD_MAX_X, WORLD_MIN_Y, WORLD_MAX_Y
    from waypoint_generation import WaypointTournament, create_default_tournament
    from waypoint_scoring import score_waypoint_list, check_waypoint_non_skippability


def load_maze_like_levels(tsv_path: str) -> List[str]:
    """Load level IDs from the maze_like_levels.tsv file."""
    level_ids = []
    try:
        with open(tsv_path, 'r') as f:
            for line in f:
                level_id = line.strip()
                if level_id and level_id.isdigit():
                    level_ids.append(level_id)
        return level_ids
    except FileNotFoundError:
        print(f"Error: Could not find {tsv_path}")
        return []


def convert_world_to_pixel_coords(waypoints: List[Dict[str, float]], 
                                 screenshot_shape: Tuple[int, int]) -> List[Dict[str, float]]:
    """Convert waypoints from world coordinates to pixel coordinates."""
    # Use coordinate transform utility for consistency
    try:
        from .coordinate_transform import world_to_pixel
    except ImportError:
        from coordinate_transform import world_to_pixel
    
    return world_to_pixel(waypoints, screenshot_shape)


def convert_pixel_to_world_coords(waypoints: List[Dict[str, float]], 
                                 screenshot_shape: Tuple[int, int]) -> List[Dict[str, float]]:
    """Convert waypoints from pixel coordinates to world coordinates."""
    # Use coordinate transform utility for consistency
    try:
        from .coordinate_transform import pixel_to_world
    except ImportError:
        from coordinate_transform import pixel_to_world
    
    return pixel_to_world(waypoints, screenshot_shape)


def create_test_screenshot(design_struct: FCDesignStruct, 
                          image_dimensions: Tuple[int, int] = (200, 145)) -> np.ndarray:
    """Create a test screenshot from design data."""
    return screenshot_design(design_struct, image_dimensions, use_rgb=False)


def validate_screenshot_for_waypoints(screenshot: np.ndarray) -> Tuple[bool, str]:
    """
    Validate that a screenshot is suitable for waypoint generation.
    
    Returns:
        Tuple of (is_valid, reason) where is_valid is True if suitable
    """
    # Check for required pixel types
    unique_pixels = np.unique(screenshot)
    
    has_source = 3 in unique_pixels
    has_sink = 4 in unique_pixels
    has_walls = 1 in unique_pixels
    
    if not has_source:
        return False, "No source pixels (type 3) found"
    
    if not has_sink:
        return False, "No sink pixels (type 4) found"
    
    # Check basic connectivity
    sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
    sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
    
    if not _check_basic_connectivity(screenshot, sources, sinks):
        return False, "Sources cannot reach sinks"
    
    return True, "Valid"


def _check_basic_connectivity(screenshot: np.ndarray, sources: List[Tuple[int, int]], 
                            sinks: List[Tuple[int, int]]) -> bool:
    """Check if sources can reach sinks using BFS."""
    from collections import deque
    
    if not sources or not sinks:
        return False
    
    height, width = screenshot.shape
    visited = set()
    queue = deque(sources)
    visited.update(sources)
    
    while queue:
        x, y = queue.popleft()
        
        if (x, y) in sinks:
            return True
        
        for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nx, ny = x + dx, y + dy
            
            if (nx < 0 or nx >= width or ny < 0 or ny >= height or 
                (nx, ny) in visited or screenshot[ny, nx] == 1):
                continue
            
            visited.add((nx, ny))
            queue.append((nx, ny))
    
    return False


def run_test_on_level_ids(level_ids: List[str], max_levels: Optional[int] = None, 
                         image_dimensions: Tuple[int, int] = (200, 145),
                         verbose: bool = True) -> Dict[str, Any]:
    """
    Run waypoint generation tests on a list of level IDs.
    
    Args:
        level_ids: List of level ID strings
        max_levels: Maximum number of levels to test (None for all)
        image_dimensions: Screenshot dimensions (width, height)
        verbose: Whether to print detailed progress
    
    Returns:
        Dictionary with test results and statistics
    """
    if get_design_by_id is None:
        raise ImportError("get_design module is required for level testing")
    
    test_cases = []
    failed_levels = []
    
    # Limit the number of levels if specified
    if max_levels:
        level_ids = level_ids[:max_levels]
    
    if verbose:
        print(f"Loading {len(level_ids)} levels...")
    
    # Load and validate levels
    for i, level_id in enumerate(level_ids):
        try:
            if verbose and (i + 1) % 10 == 0:
                print(f"  Loaded {i + 1}/{len(level_ids)} levels...")
            
            # Get design data
            design_struct = get_design_by_id(level_id)
            if design_struct is None:
                failed_levels.append((level_id, "Could not load design"))
                continue
            
            # Create screenshot
            screenshot = create_test_screenshot(design_struct, image_dimensions)
            
            # Validate screenshot
            is_valid, reason = validate_screenshot_for_waypoints(screenshot)
            if not is_valid:
                failed_levels.append((level_id, reason))
                continue
            
            test_cases.append((level_id, screenshot))
            
        except Exception as e:
            failed_levels.append((level_id, f"Error: {str(e)}"))
            continue
    
    if verbose:
        print(f"Successfully loaded {len(test_cases)} levels ({len(failed_levels)} failed)")
        if failed_levels and verbose:
            print("Failed levels:")
            for level_id, reason in failed_levels[:5]:  # Show first 5 failures
                print(f"  {level_id}: {reason}")
            if len(failed_levels) > 5:
                print(f"  ... and {len(failed_levels) - 5} more")
    
    if not test_cases:
        return {"error": "No valid test cases loaded"}
    
    # Create and run tournament
    tournament = create_default_tournament()
    results = tournament.run_tournament(test_cases, verbose=verbose)
    
    # Print final rankings
    if verbose:
        tournament.print_final_rankings()
    
    return {
        "tournament_results": results,
        "test_case_count": len(test_cases),
        "failed_levels": failed_levels,
        "image_dimensions": image_dimensions
    }


def run_quick_test(max_levels: int = 5, verbose: bool = True) -> Dict[str, Any]:
    """Run a quick test on a small number of levels."""
    # Get the path to maze_like_levels.tsv
    base_path = os.path.dirname(os.path.dirname(__file__))
    tsv_path = os.path.join(base_path, "maze_like_levels.tsv")
    
    level_ids = load_maze_like_levels(tsv_path)
    if not level_ids:
        return {"error": f"Could not load level IDs from {tsv_path}"}
    
    return run_test_on_level_ids(level_ids, max_levels=max_levels, verbose=verbose)


def create_synthetic_test_cases() -> List[Tuple[str, np.ndarray]]:
    """Create synthetic test cases for algorithm development."""
    test_cases = []
    
    # Test case 1: Simple corridor
    screenshot1 = np.zeros((50, 100), dtype=np.uint8)
    # Add walls around edges
    screenshot1[0, :] = 1
    screenshot1[-1, :] = 1  
    screenshot1[:, 0] = 1
    screenshot1[:, -1] = 1
    # Add source and sink
    screenshot1[25, 10] = 3  # source
    screenshot1[25, 90] = 4  # sink
    test_cases.append(("synthetic_corridor", screenshot1))
    
    # Test case 2: L-shaped path
    screenshot2 = np.ones((60, 60), dtype=np.uint8)  # All walls
    # Carve out L-shaped corridor
    screenshot2[10:20, 10:50] = 0  # Horizontal part
    screenshot2[10:50, 40:50] = 0  # Vertical part
    # Add source and sink
    screenshot2[15, 15] = 3  # source
    screenshot2[45, 45] = 4  # sink
    test_cases.append(("synthetic_l_shape", screenshot2))
    
    # Test case 3: Multi-path with required waypoint
    screenshot3 = np.zeros((40, 80), dtype=np.uint8)
    # Add walls to create multiple paths
    screenshot3[15:25, 20:60] = 1  # Middle barrier
    screenshot3[19, 30:50] = 0    # Gap in barrier
    # Add borders
    screenshot3[0, :] = 1
    screenshot3[-1, :] = 1
    screenshot3[:, 0] = 1
    screenshot3[:, -1] = 1
    # Add source and sink
    screenshot3[20, 5] = 3   # source
    screenshot3[20, 75] = 4  # sink
    test_cases.append(("synthetic_multipath", screenshot3))
    
    return test_cases


def test_synthetic_levels(verbose: bool = True) -> Dict[str, Any]:
    """Test waypoint generation on synthetic levels."""
    test_cases = create_synthetic_test_cases()
    
    if verbose:
        print(f"Testing {len(test_cases)} synthetic levels...")
    
    # Create and run tournament
    tournament = create_default_tournament()
    results = tournament.run_tournament(test_cases, verbose=verbose)
    
    # Print final rankings
    if verbose:
        tournament.print_final_rankings()
    
    return {
        "tournament_results": results,
        "test_case_count": len(test_cases),
        "test_type": "synthetic"
    }


if __name__ == "__main__":
    # Run quick test by default
    print("Running quick waypoint generation test...")
    
    # Try synthetic test first (doesn't require level loading)
    print("\n1. Testing synthetic levels:")
    synthetic_results = test_synthetic_levels()
    
    # Try real levels if possible
    print("\n2. Testing real levels:")
    try:
        real_results = run_quick_test(max_levels=3)
        if "error" in real_results:
            print(f"Error with real levels: {real_results['error']}")
    except Exception as e:
        print(f"Could not test real levels: {e}")
    
    print("\nTest completed!")