"""
Coordinate transformation utilities between world and pixel spaces.

Extracted from screenshot.py to provide bidirectional coordinate conversion
for waypoint generation algorithms and autotweaker integration.
"""

import numpy as np
from typing import List, Dict, Tuple

# World bounds from fcsim source (same as screenshot.py)
WORLD_MIN_X = -2000
WORLD_MAX_X = 2000
WORLD_MIN_Y = -1450
WORLD_MAX_Y = 1450

def world_to_pixel(world_coords: List[Dict[str, float]], 
                   image_dimensions: Tuple[int, int]) -> List[Dict[str, float]]:
    """
    Convert waypoints from world coordinates to pixel coordinates.
    
    Args:
        world_coords: List of waypoint dicts with 'x', 'y', 'radius' in world units
        image_dimensions: (width, height) of screenshot in pixels
        
    Returns:
        List of waypoint dicts with 'x', 'y', 'radius' in pixel coordinates
    """
    width, height = image_dimensions
    
    pixel_coords = []
    for wp in world_coords:
        # Convert position
        pixel_x = (wp['x'] - WORLD_MIN_X) * width / (WORLD_MAX_X - WORLD_MIN_X)
        pixel_y = (wp['y'] - WORLD_MIN_Y) * height / (WORLD_MAX_Y - WORLD_MIN_Y)
        
        # Convert radius (use minimum scale to maintain aspect ratio)
        scale_x = width / (WORLD_MAX_X - WORLD_MIN_X)
        scale_y = height / (WORLD_MAX_Y - WORLD_MIN_Y)
        radius_scale = min(scale_x, scale_y)
        pixel_radius = wp['radius'] * radius_scale
        
        pixel_coords.append({
            'x': float(pixel_x),
            'y': float(pixel_y), 
            'radius': float(pixel_radius)
        })
    
    return pixel_coords

def pixel_to_world(pixel_coords: List[Dict[str, float]], 
                   image_dimensions: Tuple[int, int]) -> List[Dict[str, float]]:
    """
    Convert waypoints from pixel coordinates to world coordinates.
    
    Args:
        pixel_coords: List of waypoint dicts with 'x', 'y', 'radius' in pixel units
        image_dimensions: (width, height) of screenshot in pixels
        
    Returns:
        List of waypoint dicts with 'x', 'y', 'radius' in world coordinates
    """
    width, height = image_dimensions
    
    world_coords = []
    for wp in pixel_coords:
        # Convert position (add 0.5 for pixel center, as in screenshot.py)
        world_x = WORLD_MIN_X + (wp['x'] + 0.5) * (WORLD_MAX_X - WORLD_MIN_X) / width
        world_y = WORLD_MIN_Y + (wp['y'] + 0.5) * (WORLD_MAX_Y - WORLD_MIN_Y) / height
        
        # Convert radius (use minimum scale to maintain aspect ratio)
        scale_x = (WORLD_MAX_X - WORLD_MIN_X) / width
        scale_y = (WORLD_MAX_Y - WORLD_MIN_Y) / height
        radius_scale = min(scale_x, scale_y)
        world_radius = wp['radius'] * radius_scale
        
        world_coords.append({
            'x': float(world_x),
            'y': float(world_y),
            'radius': float(world_radius)
        })
    
    return world_coords

def world_point_to_pixel(world_x: float, world_y: float, 
                        image_dimensions: Tuple[int, int]) -> Tuple[int, int]:
    """
    Convert single world coordinate point to pixel coordinates.
    
    Args:
        world_x, world_y: World coordinates
        image_dimensions: (width, height) of screenshot
        
    Returns:
        (pixel_x, pixel_y) as integers
    """
    width, height = image_dimensions
    
    pixel_x = (world_x - WORLD_MIN_X) * width / (WORLD_MAX_X - WORLD_MIN_X)
    pixel_y = (world_y - WORLD_MIN_Y) * height / (WORLD_MAX_Y - WORLD_MIN_Y)
    
    return int(pixel_x), int(pixel_y)

def pixel_point_to_world(pixel_x: int, pixel_y: int, 
                        image_dimensions: Tuple[int, int]) -> Tuple[float, float]:
    """
    Convert single pixel coordinate point to world coordinates.
    
    Args:
        pixel_x, pixel_y: Pixel coordinates
        image_dimensions: (width, height) of screenshot
        
    Returns:
        (world_x, world_y) as floats
    """
    width, height = image_dimensions
    
    world_x = WORLD_MIN_X + (pixel_x + 0.5) * (WORLD_MAX_X - WORLD_MIN_X) / width
    world_y = WORLD_MIN_Y + (pixel_y + 0.5) * (WORLD_MAX_Y - WORLD_MIN_Y) / height
    
    return float(world_x), float(world_y)

def get_world_bounds() -> Tuple[float, float, float, float]:
    """
    Get world coordinate bounds.
    
    Returns:
        (min_x, max_x, min_y, max_y) world coordinate bounds
    """
    return WORLD_MIN_X, WORLD_MAX_X, WORLD_MIN_Y, WORLD_MAX_Y

def validate_coordinate_transform(image_dimensions: Tuple[int, int]) -> bool:
    """
    Validate coordinate transformation by round-trip testing.
    
    Args:
        image_dimensions: (width, height) for transform
        
    Returns:
        True if transforms are consistent within tolerance
    """
    # Test with sample world coordinates
    test_world_coords = [
        {'x': 0.0, 'y': 0.0, 'radius': 100.0},
        {'x': -600.0, 'y': -600.0, 'radius': 300.0}, 
        {'x': 600.0, 'y': 200.0, 'radius': 50.0}
    ]
    
    # Round-trip: world -> pixel -> world
    pixel_coords = world_to_pixel(test_world_coords, image_dimensions)
    recovered_world = pixel_to_world(pixel_coords, image_dimensions)
    
    # Check tolerance (should be within 1 world unit)
    tolerance = 1.0
    
    for orig, recovered in zip(test_world_coords, recovered_world):
        if abs(orig['x'] - recovered['x']) > tolerance:
            return False
        if abs(orig['y'] - recovered['y']) > tolerance:
            return False
        if abs(orig['radius'] - recovered['radius']) > tolerance:
            return False
    
    return True

if __name__ == "__main__":
    # Test coordinate transforms
    print("Testing coordinate transforms...")
    
    # Test with common screenshot dimensions
    dimensions = [(200, 145), (400, 290), (414, 300)]
    
    for width, height in dimensions:
        print(f"\nTesting {width}x{height}:")
        
        # Test validation
        is_valid = validate_coordinate_transform((width, height))
        print(f"  Round-trip validation: {'✅' if is_valid else '❌'}")
        
        # Test example transforms
        world_waypoints = [
            {'x': -600.0, 'y': -600.0, 'radius': 300.0},
            {'x': 600.0, 'y': 200.0, 'radius': 100.0}
        ]
        
        pixel_waypoints = world_to_pixel(world_waypoints, (width, height))
        recovered_world = pixel_to_world(pixel_waypoints, (width, height))
        
        print(f"  Original world: {world_waypoints[0]}")
        print(f"  -> Pixel: {pixel_waypoints[0]}")
        print(f"  -> Recovered: {recovered_world[0]}")
    
    print("\nCoordinate transform testing complete!")