#!/usr/bin/env python3
"""
Test script for real level loading using ftlib submodule.
"""

import sys
import os

# Add ftlib test directory to path
ftlib_test_path = os.path.join(os.path.dirname(__file__), 'ftlib', 'test')
sys.path.append(ftlib_test_path)

# Add py_autotweaker to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'py_autotweaker'))

def test_get_design_import():
    """Test if we can import get_design module."""
    try:
        from get_design import FCDesignStruct, retrieveLevel, designDomToStruct
        print("✓ Successfully imported get_design module")
        return True, FCDesignStruct, retrieveLevel, designDomToStruct
    except ImportError as e:
        print(f"✗ Failed to import get_design: {e}")
        return False, None, None, None

def test_level_loading():
    """Test loading a real level."""
    success, FCDesignStruct, retrieveLevel, designDomToStruct = test_get_design_import()
    if not success:
        return False
    
    # Try to load a level from maze_like_levels.tsv
    try:
        with open('maze_like_levels.tsv', 'r') as f:
            first_level_id = f.readline().strip()
        
        print(f"Attempting to load level {first_level_id}...")
        
        # Get the XML data
        level_dom = retrieveLevel(first_level_id, is_design=False)
        if level_dom is None:
            print(f"✗ retrieveLevel returned None for level {first_level_id}")
            return False
        
        # Convert to design struct
        design = designDomToStruct(level_dom)
        if design is None:
            print(f"✗ designDomToStruct returned None for level {first_level_id}")
            return False
        
        print(f"✓ Successfully loaded level {first_level_id}")
        print(f"  Design name: {design.name}")
        print(f"  Goal pieces: {len(design.goal_pieces)}")
        print(f"  Design pieces: {len(design.design_pieces)}")
        print(f"  Level pieces: {len(design.level_pieces)}")
        print(f"  Goal area: x={design.goal_area.x}, y={design.goal_area.y}")
        
        return True, design
        
    except Exception as e:
        print(f"✗ Error loading level: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_screenshot_generation():
    """Test generating screenshot from real level data."""
    success, design = test_level_loading()
    if not success:
        return False
    
    try:
        from screenshot import screenshot_design
        
        print("Generating screenshot...")
        screenshot = screenshot_design(design, (200, 145), use_rgb=False)
        
        print(f"✓ Screenshot generated: {screenshot.shape} {screenshot.dtype}")
        print(f"  Unique pixel values: {sorted(set(screenshot.flat))}")
        
        # Check for required pixel types
        has_source = 3 in screenshot
        has_sink = 4 in screenshot
        has_walls = 1 in screenshot
        
        print(f"  Contains sources (3): {'✓' if has_source else '✗'}")
        print(f"  Contains sinks (4): {'✓' if has_sink else '✗'}")
        print(f"  Contains walls (1): {'✓' if has_walls else '✗'}")
        
        return True, screenshot
        
    except Exception as e:
        print(f"✗ Error generating screenshot: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_waypoint_generation_on_real_level():
    """Test waypoint generation on a real level."""
    success, screenshot = test_screenshot_generation()
    if not success:
        return False
    
    try:
        from waypoint_generation import create_default_tournament
        from waypoint_scoring import score_waypoint_list
        
        print("Testing waypoint generation on real level...")
        
        # Create a simple tournament
        tournament = create_default_tournament()
        
        # Test each generator
        for generator in tournament.generators[:2]:  # Just test first 2 to save time
            try:
                waypoints = generator.generate_waypoints(screenshot)
                score = score_waypoint_list(screenshot, waypoints, penalize_skippable=True)
                
                print(f"  {generator.name:15} | {len(waypoints):2d} waypoints | Score: {score:8.2f}")
                
                if waypoints:
                    print(f"    Sample waypoint: ({waypoints[0]['x']:.1f}, {waypoints[0]['y']:.1f}) r={waypoints[0]['radius']:.1f}")
                
            except Exception as e:
                print(f"  {generator.name:15} | ERROR: {e}")
        
        print("✓ Waypoint generation test completed")
        return True
        
    except Exception as e:
        print(f"✗ Error in waypoint generation: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("="*60)
    print("REAL LEVEL TESTING WITH FTLIB")
    print("="*60)
    
    print("\n1. Testing get_design import...")
    test_get_design_import()
    
    print("\n2. Testing level loading...")
    test_level_loading()
    
    print("\n3. Testing screenshot generation...")
    test_screenshot_generation()
    
    print("\n4. Testing waypoint generation...")
    test_waypoint_generation_on_real_level()
    
    print("\n" + "="*60)
    print("REAL LEVEL TEST COMPLETED")
    print("="*60)

if __name__ == "__main__":
    main()