import numpy as np
import math
from typing import Tuple

# Import types - these will be available when called from the main program
# which sets up the import path correctly
try:
    from get_design import FCDesignStruct, FCPieceStruct
except ImportError:
    # For type hints when not running from main program
    FCDesignStruct = None
    FCPieceStruct = None

# World bounds from fcsim source
WORLD_MIN_X = -2000
WORLD_MAX_X = 2000
WORLD_MIN_Y = -1450
WORLD_MAX_Y = 1450

def screenshot_design(design: FCDesignStruct, image_dimensions: Tuple[int, int], use_rgb: bool = False) -> np.ndarray:
    """
    Generate a screenshot image from design data.
    
    Args:
        design: FCDesignStruct containing design data
        image_dimensions: (width, height) of output image
        use_rgb: If True, return RGB image (Y, X, 3), else integer colors (Y, X)
    
    Returns:
        np.ndarray: Image array with piece types rendered
    """
    width, height = image_dimensions
    
    # Initialize image with background color (0)
    if use_rgb:
        image = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        image = np.zeros((height, width), dtype=np.uint8)
    
    # Calculate world-to-pixel transformation
    world_width = WORLD_MAX_X - WORLD_MIN_X
    world_height = WORLD_MAX_Y - WORLD_MIN_Y
    scale_x = width / world_width
    scale_y = height / world_height
    
    def world_to_pixel(world_x: float, world_y: float) -> Tuple[int, int]:
        """Convert world coordinates to pixel coordinates"""
        pixel_x = int((world_x - WORLD_MIN_X) * scale_x)
        pixel_y = int((world_y - WORLD_MIN_Y) * scale_y)
        return pixel_x, pixel_y
    
    def pixel_to_world(pixel_x: int, pixel_y: int) -> Tuple[float, float]:
        """Convert pixel coordinates back to world coordinates (for pixel center)"""
        world_x = WORLD_MIN_X + (pixel_x + 0.5) / scale_x
        world_y = WORLD_MIN_Y + (pixel_y + 0.5) / scale_y
        return world_x, world_y
    
    # Collect all pieces and sort by z-order (render order)
    all_pieces = []
    
    # Add all pieces with their rendering properties
    for piece in design.goal_pieces + design.design_pieces + design.level_pieces:
        all_pieces.append(piece)
    
    # Add goal area as a special piece
    goal_area_piece = FCPieceStruct(
        type_id=-1,  # Special type for goal area
        piece_id=None,
        x=design.goal_area.x,
        y=design.goal_area.y,
        w=design.goal_area.w,
        h=design.goal_area.h,
        angle=0,
        joints=[]
    )
    all_pieces.append(goal_area_piece)
    
    # Sort pieces by z-order (lower z first, higher z on top)
    def get_z_order_and_color(piece: FCPieceStruct) -> Tuple[int, int, bool, bool]:
        """Returns (z_order, color, use_best_method, use_fat_method)"""
        
        # Goal area: BEST, z between statics and dynamics (0.5), color=4
        if piece.type_id == -1:  # Goal area
            return (0, 4, True, False)  # z=0 to render under everything else
        
        # Goal rectangles and circles: FAT, z=2, color=3
        elif piece.type_id in [4, 5]:  # GP_RECT, GP_CIRC
            return (3, 3, False, True)
        
        # Static rectangles and circles: BEST, z=1, color=1 (FAT if 0 width/height/area)
        elif piece.type_id in [0, 1]:  # STATIC_RECT, STATIC_CIRC
            use_fat = (piece.w == 0 or piece.h == 0 or (piece.w * piece.h) == 0)
            return (2, 1, not use_fat, use_fat)
        
        # Dynamic rectangles and circles: BEST, z=0, color=2 (treat as static if 0 width/height/area)
        elif piece.type_id in [2, 3]:  # DYNAMIC_RECT, DYNAMIC_CIRC
            if piece.w == 0 or piece.h == 0 or (piece.w * piece.h) == 0:
                # Treat as static with FAT rendering
                return (2, 1, False, True)
            else:
                return (1, 2, True, False)
        
        # Everything else: ignore
        else:
            return (-1, 0, False, False)
    
    # Filter and sort pieces
    render_pieces = []
    for piece in all_pieces:
        z_order, color, use_best, use_fat = get_z_order_and_color(piece)
        if z_order >= 0:  # Only render non-ignored pieces
            render_pieces.append((z_order, piece, color, use_best, use_fat))
    
    # Sort by z-order (render lower z first)
    render_pieces.sort(key=lambda x: x[0])
    
    # Render each piece
    for z_order, piece, color, use_best, use_fat in render_pieces:
        is_circle = piece.type_id in [1, 3, 5]  # STATIC_CIRC, DYNAMIC_CIRC, GP_CIRC
        
        if is_circle:
            _render_circle(image, piece, color, use_best, use_fat, scale_x, scale_y, use_rgb)
        else:
            _render_rectangle(image, piece, color, use_best, use_fat, scale_x, scale_y, use_rgb)
    
    # Convert to RGB if requested
    if use_rgb:
        _convert_to_rgb(image)
    
    return image

def _render_circle(image: np.ndarray, piece: FCPieceStruct, color: int, use_best: bool, use_fat: bool, 
                   scale_x: float, scale_y: float, use_rgb: bool):
    """Render a circle piece to the image"""
    height, width = image.shape[:2]
    
    # Circle parameters
    center_x, center_y = piece.x, piece.y
    radius = piece.w / 2.0  # Assuming w is diameter
    
    # Convert to pixel coordinates
    center_px = int((center_x - WORLD_MIN_X) * scale_x)
    center_py = int((center_y - WORLD_MIN_Y) * scale_y)
    radius_px = radius * scale_x  # Use x scale for radius
    
    if use_fat:
        # FAT method: pixel colored if circle overlaps it at all
        # Ensure at least 1 pixel is colored
        radius_px = max(radius_px, 0.5)
        
        # Find bounding box
        min_px = max(0, int(center_px - radius_px - 1))
        max_px = min(width, int(center_px + radius_px + 2))
        min_py = max(0, int(center_py - radius_px - 1))
        max_py = min(height, int(center_py + radius_px + 2))
        
        for py in range(min_py, max_py):
            for px in range(min_px, max_px):
                # Check if pixel overlaps with circle
                dx = px - center_px
                dy = py - center_py
                if dx*dx + dy*dy <= (radius_px + 0.5)**2:
                    _set_pixel(image, py, px, color, use_rgb)
    
    else:
        # BEST method: pixel colored based on its center
        # Find bounding box
        min_px = max(0, int(center_px - radius_px))
        max_px = min(width, int(center_px + radius_px + 1))
        min_py = max(0, int(center_py - radius_px))
        max_py = min(height, int(center_py + radius_px + 1))
        
        for py in range(min_py, max_py):
            for px in range(min_px, max_px):
                # Check if pixel center is inside circle
                dx = px + 0.5 - center_px
                dy = py + 0.5 - center_py
                if dx*dx + dy*dy <= radius_px*radius_px:
                    _set_pixel(image, py, px, color, use_rgb)

def _render_rectangle(image: np.ndarray, piece: FCPieceStruct, color: int, use_best: bool, use_fat: bool,
                      scale_x: float, scale_y: float, use_rgb: bool):
    """Render a rectangle piece to the image"""
    height, width = image.shape[:2]
    
    # Rectangle parameters
    center_x, center_y = piece.x, piece.y
    half_width = piece.w / 2.0
    half_height = piece.h / 2.0
    angle = piece.angle
    
    if use_fat:
        # FAT method: ensure at least 1 pixel, check overlap
        # For simplicity, use axis-aligned bounding box approach
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Find rotated rectangle corners
        corners = []
        for dx, dy in [(-half_width, -half_height), (half_width, -half_height), 
                       (half_width, half_height), (-half_width, half_height)]:
            rot_x = center_x + dx * cos_a - dy * sin_a
            rot_y = center_y + dx * sin_a + dy * cos_a
            corners.append((rot_x, rot_y))
        
        # Find bounding box in world coordinates
        min_x = min(c[0] for c in corners)
        max_x = max(c[0] for c in corners)
        min_y = min(c[1] for c in corners)
        max_y = max(c[1] for c in corners)
        
        # Convert to pixel coordinates with expansion
        min_px = max(0, int((min_x - WORLD_MIN_X) * scale_x) - 1)
        max_px = min(width, int((max_x - WORLD_MIN_X) * scale_x) + 2)
        min_py = max(0, int((min_y - WORLD_MIN_Y) * scale_y) - 1)
        max_py = min(height, int((max_y - WORLD_MIN_Y) * scale_y) + 2)
        
        # Ensure at least 1 pixel
        if max_px <= min_px:
            max_px = min_px + 1
        if max_py <= min_py:
            max_py = min_py + 1
        
        for py in range(min_py, max_py):
            for px in range(min_px, max_px):
                # Check if pixel overlaps with rectangle
                if _point_in_rotated_rect(px, py, center_x, center_y, half_width, half_height, 
                                         angle, scale_x, scale_y, fat=True):
                    _set_pixel(image, py, px, color, use_rgb)
    
    else:
        # BEST method: pixel colored based on center
        # Find bounding box
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        
        # Find rotated rectangle corners
        corners = []
        for dx, dy in [(-half_width, -half_height), (half_width, -half_height), 
                       (half_width, half_height), (-half_width, half_height)]:
            rot_x = center_x + dx * cos_a - dy * sin_a
            rot_y = center_y + dx * sin_a + dy * cos_a
            corners.append((rot_x, rot_y))
        
        # Find bounding box in world coordinates
        min_x = min(c[0] for c in corners)
        max_x = max(c[0] for c in corners)
        min_y = min(c[1] for c in corners)
        max_y = max(c[1] for c in corners)
        
        # Convert to pixel coordinates
        min_px = max(0, int((min_x - WORLD_MIN_X) * scale_x))
        max_px = min(width, int((max_x - WORLD_MIN_X) * scale_x) + 1)
        min_py = max(0, int((min_y - WORLD_MIN_Y) * scale_y))
        max_py = min(height, int((max_y - WORLD_MIN_Y) * scale_y) + 1)
        
        for py in range(min_py, max_py):
            for px in range(min_px, max_px):
                # Check if pixel center is inside rectangle
                if _point_in_rotated_rect(px + 0.5, py + 0.5, center_x, center_y, half_width, half_height,
                                         angle, scale_x, scale_y, fat=False):
                    _set_pixel(image, py, px, color, use_rgb)

def _point_in_rotated_rect(px: float, py: float, center_x: float, center_y: float, 
                          half_width: float, half_height: float, angle: float,
                          scale_x: float, scale_y: float, fat: bool) -> bool:
    """Check if a pixel point is inside a rotated rectangle"""
    # Convert pixel to world coordinates
    world_x = WORLD_MIN_X + px / scale_x
    world_y = WORLD_MIN_Y + py / scale_y
    
    # Translate to rectangle center
    dx = world_x - center_x
    dy = world_y - center_y
    
    # Rotate to rectangle's local coordinates
    cos_a = math.cos(-angle)
    sin_a = math.sin(-angle)
    local_x = dx * cos_a - dy * sin_a
    local_y = dx * sin_a + dy * cos_a
    
    # Check if inside rectangle
    if fat:
        # FAT: allow some overlap
        tolerance = 0.5 / min(scale_x, scale_y)  # Half pixel in world units
        return (abs(local_x) <= half_width + tolerance and 
                abs(local_y) <= half_height + tolerance)
    else:
        # BEST: exact boundary
        return abs(local_x) <= half_width and abs(local_y) <= half_height

def _set_pixel(image: np.ndarray, py: int, px: int, color: int, use_rgb: bool):
    """Set a pixel to the given color"""
    if 0 <= py < image.shape[0] and 0 <= px < image.shape[1]:
        if use_rgb:
            # Will be converted to RGB later, just set integer color for now
            image[py, px, 0] = color  # Store color in red channel temporarily
        else:
            image[py, px] = color

def _convert_to_rgb(image: np.ndarray):
    """Convert integer colors to RGB colors in-place"""
    # Color mappings based on fcsim color table and requirements
    color_map = {
        0: (135, 189, 241),  # Sky blue background (0xff87bdf1)
        1: (1, 190, 2),      # Green static color (0xff01be02)  
        2: (249, 137, 49),   # Orange dynamic color (0xfff98931)
        3: (254, 103, 102),  # Pink goal color (0xfffe6766)
        4: (188, 102, 103),  # Goal area color (0xffbc6667)
    }
    
    height, width = image.shape[:2]
    
    # Create new RGB image
    rgb_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    for color_id, (r, g, b) in color_map.items():
        mask = image[:, :, 0] == color_id
        rgb_image[mask] = [r, g, b]
    
    # Copy back to original image
    image[:] = rgb_image[:]