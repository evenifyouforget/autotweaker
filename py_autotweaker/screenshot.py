import numpy as np
import math
import warnings
from typing import Tuple, List
from get_design import FCDesignStruct, FCPieceStruct
from PIL import Image, ImageDraw, ImageFont

# World bounds from fcsim source
WORLD_MIN_X = -2000
WORLD_MAX_X = 2000
WORLD_MIN_Y = -1450
WORLD_MAX_Y = 1450

def screenshot_design(design: FCDesignStruct, image_dimensions: Tuple[int, int], use_rgb: bool = False) -> np.ndarray:
    """
    Generate a screenshot image from design data using vectorized numpy operations.
    
    Args:
        design: FCDesignStruct containing design data
        image_dimensions: (width, height) of output image
        use_rgb: If True, return RGB image (Y, X, 3), else integer colors (Y, X)
    
    Returns:
        np.ndarray: Image array with piece types rendered
    """
    width, height = image_dimensions
    
    # Check for severely deformed aspect ratios
    world_width = WORLD_MAX_X - WORLD_MIN_X
    world_height = WORLD_MAX_Y - WORLD_MIN_Y
    world_aspect_ratio = world_width / world_height
    
    # Calculate corrected dimensions
    corrected_width_for_height = int(height * world_aspect_ratio + 0.5)
    corrected_height_for_width = int(width / world_aspect_ratio + 0.5)
    
    # Check if aspect ratio is off by more than a few pixels
    width_diff = abs(width - corrected_width_for_height)
    height_diff = abs(height - corrected_height_for_width)
    
    # Warn if it is off by 1 pixel in both ways (likely a mistake)
    if width_diff > 1 and height_diff > 1:
        warnings.warn(
            f"Image dimensions {width}x{height} have inaccurate aspect ratio. "
            f"World aspect ratio is {world_aspect_ratio:.3f}. "
            f"For height {height}, recommended width is {corrected_width_for_height}. "
            f"For width {width}, recommended height is {corrected_height_for_width}.",
            UserWarning
        )
    
    # Initialize image with background color (0)
    image = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate world-to-pixel transformation
    scale_x = width / world_width
    scale_y = height / world_height
    
    # Create pixel coordinate grids
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    # Convert to world coordinates (pixel centers)
    world_x = WORLD_MIN_X + (x_coords + 0.5) / scale_x
    world_y = WORLD_MIN_Y + (y_coords + 0.5) / scale_y
    
    # Collect all pieces with their rendering properties
    render_data = []
    
    # Add goal area first (z=0, renders under everything)
    goal_area = {
        'x': design.goal_area.x, 'y': design.goal_area.y,
        'w': design.goal_area.w, 'h': design.goal_area.h,
        'angle': 0, 'is_circle': False, 'z_order': 0, 'color': 4,
        'use_fat': False
    }
    render_data.append(goal_area)
    
    # Process all pieces
    all_pieces = design.goal_pieces + design.design_pieces + design.level_pieces
    
    for piece in all_pieces:
        z_order, color, use_best, use_fat = _get_render_props(piece)
        if z_order >= 0:  # Only render non-ignored pieces
            piece_data = {
                'x': piece.x, 'y': piece.y, 'w': piece.w, 'h': piece.h,
                'angle': piece.angle, 'is_circle': piece.type_id in [1, 3, 5],
                'z_order': z_order, 'color': color, 'use_fat': use_fat
            }
            render_data.append(piece_data)
    
    # Sort by z-order (render lower z first)
    render_data.sort(key=lambda x: x['z_order'])
    
    # Render each piece using vectorized operations
    for piece_data in render_data:
        if piece_data['is_circle']:
            _render_circle_vectorized(image, world_x, world_y, piece_data)
        else:
            _render_rectangle_vectorized(image, world_x, world_y, piece_data)
    
    # Convert to RGB if requested
    if use_rgb:
        return _convert_to_rgb_vectorized(image)
    
    return image

def _get_render_props(piece: FCPieceStruct) -> Tuple[int, int, bool, bool]:
    """Get rendering properties for a piece"""
    # Goal rectangles and circles: FAT, z=3, color=3
    if piece.type_id in [4, 5]:  # GP_RECT, GP_CIRC
        return (3, 3, False, True)
    
    # Static rectangles and circles: BEST, z=2, color=1 (FAT if 0 area)
    elif piece.type_id in [0, 1]:  # STATIC_RECT, STATIC_CIRC
        use_fat = (piece.w == 0 or piece.h == 0 or (piece.w * piece.h) == 0)
        return (2, 1, not use_fat, use_fat)
    
    # Dynamic rectangles and circles: BEST, z=1, color=2 (treat as static if 0 area)
    elif piece.type_id in [2, 3]:  # DYNAMIC_RECT, DYNAMIC_CIRC
        if piece.w == 0 or piece.h == 0 or (piece.w * piece.h) == 0:
            # Treat as static with FAT rendering
            return (2, 1, False, True)
        else:
            return (1, 2, True, False)
    
    # Everything else: ignore
    else:
        return (-1, 0, False, False)

def _render_circle_vectorized(image: np.ndarray, world_x: np.ndarray, world_y: np.ndarray, circle_data: dict):
    """Render a circle using vectorized numpy operations"""
    center_x = circle_data['x']
    center_y = circle_data['y']
    radius = circle_data['w'] / 2.0
    color = circle_data['color']
    use_fat = circle_data['use_fat']
    
    # Calculate distance from center for all pixels
    dx = world_x - center_x
    dy = world_y - center_y
    distances_sq = dx*dx + dy*dy
    radius_sq = radius*radius
    
    if use_fat:
        # FAT method: ensure at least some pixels are colored
        # Expand radius slightly to ensure visibility
        effective_radius_sq = max(radius_sq, 1.0)  # Minimum radius in world units
        mask = distances_sq <= effective_radius_sq
    else:
        # BEST method: exact circle boundary
        mask = distances_sq <= radius_sq
    
    # Apply color where mask is True
    image[mask] = color

def _render_rectangle_vectorized(image: np.ndarray, world_x: np.ndarray, world_y: np.ndarray, rect_data: dict):
    """Render a rectangle using vectorized numpy operations"""
    center_x = rect_data['x']
    center_y = rect_data['y'] 
    half_width = rect_data['w'] / 2.0
    half_height = rect_data['h'] / 2.0
    angle = rect_data['angle']
    color = rect_data['color']
    use_fat = rect_data['use_fat']
    
    # Translate to rectangle center
    dx = world_x - center_x
    dy = world_y - center_y
    
    if angle != 0:
        # Rotate to rectangle's local coordinates
        cos_a = math.cos(-angle)
        sin_a = math.sin(-angle)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
    else:
        # No rotation needed
        local_x = dx
        local_y = dy
    
    if use_fat:
        # FAT method: ensure visibility with tolerance
        tolerance = 1.0  # World units
        x_mask = np.abs(local_x) <= (half_width + tolerance)
        y_mask = np.abs(local_y) <= (half_height + tolerance)
    else:
        # BEST method: exact boundary
        x_mask = np.abs(local_x) <= half_width
        y_mask = np.abs(local_y) <= half_height
    
    # Combine masks and apply color
    mask = x_mask & y_mask
    image[mask] = color

def _convert_to_rgb_vectorized(image: np.ndarray) -> np.ndarray:
    """Convert integer colors to RGB using vectorized operations"""
    height, width = image.shape
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Color mappings based on fcsim color table
    color_map = {
        0: [135, 189, 241],  # Sky blue background (0xff87bdf1)
        1: [1, 190, 2],      # Green static color (0xff01be02)  
        2: [249, 137, 49],   # Orange dynamic color (0xfff98931)
        3: [254, 103, 102],  # Pink goal color (0xfffe6766)
        4: [188, 102, 103],  # Goal area color (0xffbc6667)
    }
    
    # Apply colors using vectorized operations
    for color_id, rgb in color_map.items():
        mask = image == color_id
        rgb_image[mask] = rgb
    
    return rgb_image

def upscale_image(image: np.ndarray, scale_factor: int) -> np.ndarray:
    """
    Upscale an image by an integer factor using nearest neighbor scaling.
    
    Args:
        image: Input image array, either (H, W) for grayscale or (H, W, 3) for RGB
        scale_factor: Integer scaling factor (e.g., 4 for 4x upscaling)
    
    Returns:
        np.ndarray: Upscaled image with pixel art effect
    """
    if not isinstance(scale_factor, int) or scale_factor < 1:
        raise ValueError("scale_factor must be a positive integer")
    
    if scale_factor == 1:
        return image.copy()
    
    return np.repeat(np.repeat(image, scale_factor, axis=0), scale_factor, axis=1)

def draw_waypoints_preview(rgb_image: np.ndarray, waypoints: List, 
                          design_struct: FCDesignStruct = None) -> Image.Image:
    """
    Draw waypoints preview on an RGB image with path visualization.
    
    Args:
        rgb_image: RGB image array (H, W, 3)
        waypoints: List of waypoint dictionaries with keys 'x', 'y', 'radius' (in world coordinates)
        design_struct: FCDesignStruct containing goal pieces and goal area (optional, for path drawing)
    
    Returns:
        PIL.Image: Image with waypoints and path overlaid
    """
    if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
        raise ValueError("rgb_image must be a 3-channel RGB image (H, W, 3)")
    
    # Convert numpy array to PIL Image
    pil_image = Image.fromarray(rgb_image.astype(np.uint8))
    draw = ImageDraw.Draw(pil_image)
    
    # Image dimensions
    height, width = rgb_image.shape[:2]
    
    # World to pixel coordinate conversion
    def world_to_pixel(world_x, world_y):
        pixel_x = (world_x - WORLD_MIN_X) * width / (WORLD_MAX_X - WORLD_MIN_X)
        pixel_y = (world_y - WORLD_MIN_Y) * height / (WORLD_MAX_Y - WORLD_MIN_Y)
        return int(pixel_x), int(pixel_y)
    
    # Define colors
    waypoint_color = (128, 128, 128)  # Joint gray
    path_color = (96, 96, 96)  # Darker gray for path
    text_color = (255, 255, 255)  # White text
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 12)
    except (OSError, IOError):
        try:
            font = ImageFont.load_default()
        except:
            font = None
    
    # Extract goal pieces and goal area from design_struct
    goal_pieces_coords = []
    goal_area_center = None
    
    if design_struct:
        # Extract goal piece positions
        for piece in design_struct.goal_pieces:
            goal_pieces_coords.append((piece.x, piece.y))
        
        # Extract goal area center
        goal_area_center = (design_struct.goal_area.x, design_struct.goal_area.y)
    
    # Convert waypoint positions to pixels
    waypoint_pixels = []
    for wp in waypoints:
        wp_pixel = world_to_pixel(wp["x"], wp["y"])
        waypoint_pixels.append(wp_pixel)
    
    # Build the path: goal_pieces -> wp1 -> wp2 -> ... -> wpN -> goal_area
    
    # Create the full path nodes: [goal_area] + waypoints (reversed) 
    path_nodes = []
    if goal_area_center:
        path_nodes.append(world_to_pixel(goal_area_center[0], goal_area_center[1]))
    
    # Add waypoints in reverse order (so we can walk backwards)
    for wp_pixel in reversed(waypoint_pixels):
        path_nodes.append(wp_pixel)
    
    # Draw the sequential path through waypoints to goal
    if len(path_nodes) > 1:
        for i in range(len(path_nodes) - 1):
            start_point = path_nodes[i + 1]  # Start from later waypoint
            end_point = path_nodes[i]        # Go to earlier waypoint/goal
            draw.line([start_point, end_point], fill=path_color, width=2)
    
    # Draw lines from each goal piece to the first waypoint (or goal if no waypoints)
    if goal_pieces_coords:
        # Determine the target: first waypoint if waypoints exist, otherwise goal area
        if waypoint_pixels:
            target_pixel = waypoint_pixels[0]  # First waypoint
        elif goal_area_center:
            target_pixel = world_to_pixel(goal_area_center[0], goal_area_center[1])  # Goal area
        else:
            target_pixel = None
        
        # Draw line from each goal piece to the target
        if target_pixel:
            for gp_x, gp_y in goal_pieces_coords:
                gp_pixel = world_to_pixel(gp_x, gp_y)
                draw.line([gp_pixel, target_pixel], fill=path_color, width=2)
    
    # Draw waypoints as circles with numbers
    for i, (wp, (px, py)) in enumerate(zip(waypoints, waypoint_pixels)):
        # Convert radius from world coordinates to pixels
        radius_world = wp["radius"]
        radius_pixels = radius_world * min(width / (WORLD_MAX_X - WORLD_MIN_X), 
                                          height / (WORLD_MAX_Y - WORLD_MIN_Y))
        radius_pixels = max(5, int(radius_pixels))  # Minimum visible radius
        
        # Draw circle outline
        bbox = [px - radius_pixels, py - radius_pixels, 
                px + radius_pixels, py + radius_pixels]
        draw.ellipse(bbox, outline=waypoint_color, width=2)
        
        # Draw waypoint number
        waypoint_number = str(i + 1)
        if font:
            # Get text bounding box for centering
            bbox = draw.textbbox((0, 0), waypoint_number, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        else:
            # Rough estimation for default font
            text_width = len(waypoint_number) * 6
            text_height = 10
        
        text_x = px - text_width // 2
        text_y = py - text_height // 2
        
        # Draw text background circle for better visibility
        text_bg_radius = max(text_width, text_height) // 2 + 2
        text_bg_bbox = [px - text_bg_radius, py - text_bg_radius,
                        px + text_bg_radius, py + text_bg_radius]
        draw.ellipse(text_bg_bbox, fill=(0, 0, 0, 128))  # Semi-transparent black
        
        if font:
            draw.text((text_x, text_y), waypoint_number, fill=text_color, font=font)
        else:
            draw.text((text_x, text_y), waypoint_number, fill=text_color)
    
    # Note: Goal pieces and goal area are already visible in the RGB image,
    # so we don't need to draw additional markers for them
    
    return pil_image