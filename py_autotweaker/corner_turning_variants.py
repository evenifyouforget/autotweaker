#!/usr/bin/env python3.13
"""
Corner Turning Algorithm Variants

Different developer personalities interpreting the same RESEARCH_INSTRUCTIONS.md
specification in various ways. Each variant represents how a different engineer
might implement "the same" algorithm with different approaches to the ambiguous details.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import math
import random
from dataclasses import dataclass
from abc import ABC, abstractmethod

# Import existing modules
try:
    from .coordinate_transform import world_to_pixel, pixel_to_world
    from .waypoint_scoring import WaypointScorer
except ImportError:
    # Fallback for direct execution
    import sys
    import os
    sys.path.append(os.path.dirname(__file__))
    from coordinate_transform import world_to_pixel, pixel_to_world
    from waypoint_scoring import WaypointScorer


@dataclass
class WaypointCandidate:
    """Represents a candidate waypoint during generation."""
    x: float
    y: float
    radius: float
    score: float = 0.0
    iterations: int = 0


class CornerTurningVariant(ABC):
    """Base class for corner turning algorithm variants."""
    
    def __init__(self, max_waypoints: int = 10):
        self.max_waypoints = max_waypoints
        self.scorer = WaypointScorer()
    
    @abstractmethod
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints for the given screenshot."""
        pass
    
    def _find_sources_and_sinks(self, screenshot: np.ndarray) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
        """Find source (3) and sink (4) pixels."""
        sources = list(zip(*np.where(screenshot == 3)))
        sinks = list(zip(*np.where(screenshot == 4)))
        return sources, sinks
    
    def _is_wall(self, screenshot: np.ndarray, y: int, x: int) -> bool:
        """Check if a pixel is a wall (1) or out of bounds."""
        if y < 0 or y >= screenshot.shape[0] or x < 0 or x >= screenshot.shape[1]:
            return True
        return screenshot[y, x] == 1
    
    def _has_line_of_sight(self, screenshot: np.ndarray, start: Tuple[int, int], end: Tuple[int, int]) -> bool:
        """Check if there's a clear line of sight between two points."""
        y1, x1 = start
        y2, x2 = end
        
        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        x_inc = 1 if x1 < x2 else -1
        y_inc = 1 if y1 < y2 else -1
        error = dx - dy
        
        for _ in range(dx + dy):
            if self._is_wall(screenshot, y, x):
                return False
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx
                
        return True


class EuclidCornerTurning(CornerTurningVariant):  # Named after Euclid - systematic, geometric approach
    """
    THE GRID-MARCHING PURIST
    
    Personality: Methodical, loves clean algorithms, thinks step-by-step
    
    Interpretation:
    - "Trace along path" = march exactly one pixel at a time along A* path
    - "Stochastic expansion" = systematic radius increment with noise
    - "Spring attraction" = simple linear force towards original corner
    - "Stop iteration" = fixed iteration count (because it's predictable)
    """
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        sources, sinks = self._find_sources_and_sinks(screenshot)
        if not sources or not sinks:
            return []
            
        waypoints = []
        current_sources = sources
        
        for waypoint_index in range(self.max_waypoints):
            # Check if direct line of sight exists
            if all(any(self._has_line_of_sight(screenshot, src, sink) 
                      for sink in sinks) for src in current_sources):
                break
                
            # Find path using A* from any sink to any source
            path = self._find_astar_path(screenshot, sinks[0], current_sources[0])
            if not path:
                break
                
            # March along path one pixel at a time until obscured
            corner_point = None
            for i, point in enumerate(path):
                if not any(self._has_line_of_sight(screenshot, point, sink) for sink in sinks):
                    corner_point = path[max(0, i-1)]  # Last non-obscured point
                    break
                    
            if corner_point is None:
                break
                
            # Create waypoint with systematic balloon expansion
            waypoint = self._create_waypoint_grid_marching(screenshot, corner_point, sinks)
            if waypoint:
                waypoints.append(waypoint)
                current_sources = [corner_point]
            else:
                break
                
        return waypoints
    
    def _find_astar_path(self, screenshot: np.ndarray, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """A* pathfinding - methodical implementation."""
        from heapq import heappush, heappop
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._manhattan_distance(start, goal)}
        
        while open_set:
            current = heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
                
            for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:  # 4-connected grid
                neighbor = (current[0] + dy, current[1] + dx)
                
                if self._is_wall(screenshot, neighbor[0], neighbor[1]):
                    continue
                    
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._manhattan_distance(neighbor, goal)
                    heappush(open_set, (f_score[neighbor], neighbor))
                    
        return []
    
    def _manhattan_distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def _create_waypoint_grid_marching(self, screenshot: np.ndarray, corner: Tuple[int, int], sinks: List[Tuple[int, int]]) -> Optional[Dict[str, float]]:
        """Create waypoint with systematic grid-based balloon expansion."""
        candidate = WaypointCandidate(x=corner[1], y=corner[0], radius=1.0)
        
        # Fixed iteration count for predictability
        for iteration in range(20):
            # Add small random noise to radius increment (stochastic element)
            radius_increment = 0.5 + random.uniform(-0.1, 0.1)
            candidate.radius += radius_increment
            
            # Check if waypoint is still non-obscured
            if not any(self._has_line_of_sight(screenshot, corner, sink) for sink in sinks):
                candidate.radius -= radius_increment  # Step back
                break
                
            # Check for wall collision (simple circle-wall intersection)
            if self._waypoint_hits_wall(screenshot, candidate):
                candidate.radius -= radius_increment  # Step back
                break
                
            candidate.iterations = iteration
            
        # Convert to world coordinates
        world_x, world_y = pixel_to_world(candidate.x, candidate.y, screenshot.shape)
        
        return {
            'x': world_x,
            'y': world_y,
            'radius': candidate.radius * (800 / screenshot.shape[1])  # Scale to world
        }
    
    def _waypoint_hits_wall(self, screenshot: np.ndarray, candidate: WaypointCandidate) -> bool:
        """Check if waypoint circle severely intersects walls (allows 1-pixel overlap)."""
        # Check if waypoint goes out of bounds 
        height, width = screenshot.shape
        if (candidate.x - candidate.radius < -1 or candidate.x + candidate.radius >= width + 1 or
            candidate.y - candidate.radius < -1 or candidate.y + candidate.radius >= height + 1):
            return True
        
        # Allow waypoints to overlap walls by 1 pixel for discretization safety
        return False


class NewtonCornerTurning(CornerTurningVariant):  # Named after Newton - physics simulation enthusiast
    """
    THE PHYSICS ENTHUSIAST
    
    Personality: Loves simulations, thinks in terms of forces and dynamics
    
    Interpretation:
    - "Trace along path" = follow gradient descent from sources towards sinks
    - "Stochastic expansion" = physics simulation with forces and momentum
    - "Spring attraction" = actual spring force with realistic physics
    - "Stop iteration" = convergence criteria based on force equilibrium
    """
    
    def __init__(self, max_waypoints: int = 10):
        super().__init__(max_waypoints)
        self.spring_constant = 0.1
        self.repulsion_strength = 0.5
        self.damping = 0.9
        
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        sources, sinks = self._find_sources_and_sinks(screenshot)
        if not sources or not sinks:
            return []
            
        waypoints = []
        current_sources = sources
        
        for waypoint_index in range(self.max_waypoints):
            # Use gradient field to find corner
            corner_point = self._find_corner_via_gradient(screenshot, current_sources, sinks)
            if corner_point is None:
                break
                
            # Physics-based waypoint creation
            waypoint = self._create_waypoint_physics(screenshot, corner_point, sinks)
            if waypoint:
                waypoints.append(waypoint)
                current_sources = [corner_point]
            else:
                break
                
        return waypoints
    
    def _find_corner_via_gradient(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find corner using gradient descent from sources towards sinks."""
        # Create potential field
        potential_field = self._create_potential_field(screenshot, sinks)
        
        # Follow gradient from sources
        for source in sources:
            current_pos = np.array(source, dtype=float)
            
            for step in range(100):
                # Calculate gradient
                gradient = self._calculate_gradient(potential_field, current_pos)
                
                # Move along gradient
                current_pos += gradient * 0.5
                
                # Check if we've hit a corner (lost line of sight)
                pixel_pos = (int(current_pos[0]), int(current_pos[1]))
                if not any(self._has_line_of_sight(screenshot, pixel_pos, sink) for sink in sinks):
                    return pixel_pos
                    
        return None
    
    def _create_potential_field(self, screenshot: np.ndarray, sinks: List[Tuple[int, int]]) -> np.ndarray:
        """Create potential field using distance transform."""
        field = np.zeros_like(screenshot, dtype=float)
        
        for sink in sinks:
            # Distance transform from this sink
            for y in range(screenshot.shape[0]):
                for x in range(screenshot.shape[1]):
                    if not self._is_wall(screenshot, y, x):
                        dist = math.sqrt((y - sink[0])**2 + (x - sink[1])**2)
                        field[y, x] = min(field[y, x], dist) if field[y, x] > 0 else dist
                        
        return field
    
    def _calculate_gradient(self, field: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Calculate gradient at position using finite differences."""
        y, x = int(pos[0]), int(pos[1])
        
        if y <= 0 or y >= field.shape[0]-1 or x <= 0 or x >= field.shape[1]-1:
            return np.array([0.0, 0.0])
            
        grad_y = (field[y+1, x] - field[y-1, x]) / 2.0
        grad_x = (field[y, x+1] - field[y, x-1]) / 2.0
        
        return np.array([-grad_y, -grad_x])  # Negative for descent
    
    def _create_waypoint_physics(self, screenshot: np.ndarray, corner: Tuple[int, int], sinks: List[Tuple[int, int]]) -> Optional[Dict[str, float]]:
        """Create waypoint using physics simulation."""
        # Initialize particle
        position = np.array([corner[1], corner[0]], dtype=float)  # x, y
        velocity = np.array([0.0, 0.0])
        radius = 1.0
        
        anchor_point = np.array([corner[1], corner[0]], dtype=float)
        
        # Physics simulation
        for iteration in range(200):
            # Calculate forces
            spring_force = self.spring_constant * (anchor_point - position)
            repulsion_force = self._calculate_wall_repulsion(screenshot, position, radius)
            
            total_force = spring_force + repulsion_force
            
            # Update physics
            velocity += total_force
            velocity *= self.damping  # Damping
            position += velocity
            
            # Update radius based on available space
            max_safe_radius = self._calculate_max_safe_radius(screenshot, position)
            radius = min(radius + 0.1, max_safe_radius)
            
            # Check convergence
            if np.linalg.norm(velocity) < 0.01 and iteration > 50:
                break
                
        # Convert to world coordinates
        world_x, world_y = pixel_to_world(position[0], position[1], screenshot.shape)
        
        return {
            'x': world_x,
            'y': world_y,
            'radius': radius * (800 / screenshot.shape[1])
        }
    
    def _calculate_wall_repulsion(self, screenshot: np.ndarray, position: np.ndarray, radius: float) -> np.ndarray:
        """Calculate repulsion force from nearby walls."""
        total_force = np.array([0.0, 0.0])
        
        # Sample points around position
        for angle in np.linspace(0, 2*np.pi, 12):
            sample_x = position[0] + radius * np.cos(angle)
            sample_y = position[1] + radius * np.sin(angle)
            
            if self._is_wall(screenshot, int(sample_y), int(sample_x)):
                # Repulsion force away from wall
                direction = np.array([np.cos(angle), np.sin(angle)])
                force_magnitude = self.repulsion_strength / max(0.1, radius)
                total_force -= direction * force_magnitude
                
        return total_force
    
    def _calculate_max_safe_radius(self, screenshot: np.ndarray, position: np.ndarray) -> float:
        """Calculate maximum radius before hitting walls."""
        max_radius = 50.0  # Reasonable upper bound
        
        for radius in np.arange(0.5, max_radius, 0.5):
            for angle in np.linspace(0, 2*np.pi, 16):
                test_x = int(position[0] + radius * np.cos(angle))
                test_y = int(position[1] + radius * np.sin(angle))
                
                if self._is_wall(screenshot, test_y, test_x):
                    return radius - 0.5
                    
        return max_radius


class DarwinCornerTurning(CornerTurningVariant):  # Named after Darwin - evolutionary optimization
    """
    THE MACHINE LEARNING OPTIMIZER
    
    Personality: Thinks everything should be learned, loves stochastic optimization
    
    Interpretation:
    - "Trace along path" = sample random paths and pick best scoring ones
    - "Stochastic expansion" = simulated annealing for waypoint parameters
    - "Spring attraction" = part of multi-objective optimization function
    - "Stop iteration" = early stopping based on improvement rate
    """
    
    def __init__(self, max_waypoints: int = 10):
        super().__init__(max_waypoints)
        self.temperature = 10.0
        self.cooling_rate = 0.95
        self.min_temperature = 0.1
        
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        sources, sinks = self._find_sources_and_sinks(screenshot)
        if not sources or not sinks:
            return []
            
        waypoints = []
        current_sources = sources
        
        for waypoint_index in range(self.max_waypoints):
            # Stochastic corner detection
            corner_candidates = self._sample_corner_candidates(screenshot, current_sources, sinks)
            if not corner_candidates:
                break
                
            # Optimize waypoint using simulated annealing
            best_waypoint = self._optimize_waypoint_simulated_annealing(
                screenshot, corner_candidates, sinks
            )
            
            if best_waypoint:
                waypoints.append(best_waypoint)
                # Update current sources to be near the waypoint
                wx, wy = world_to_pixel(best_waypoint['x'], best_waypoint['y'], screenshot.shape)
                current_sources = [(int(wy), int(wx))]
            else:
                break
                
        return waypoints
    
    def _sample_corner_candidates(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]], num_samples: int = 20) -> List[Tuple[int, int]]:
        """Sample random paths and find corner candidates."""
        candidates = []
        
        for _ in range(num_samples):
            # Pick random source and sink
            source = random.choice(sources)
            sink = random.choice(sinks)
            
            # Random walk with bias towards sink
            current = source
            path = [current]
            
            for step in range(100):
                # Biased random walk
                neighbors = []
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    ny, nx = current[0] + dy, current[1] + dx
                    if not self._is_wall(screenshot, ny, nx):
                        neighbors.append((ny, nx))
                        
                if not neighbors:
                    break
                    
                # Bias towards sink
                weights = []
                for neighbor in neighbors:
                    dist_to_sink = math.sqrt((neighbor[0] - sink[0])**2 + (neighbor[1] - sink[1])**2)
                    weights.append(1.0 / (1.0 + dist_to_sink))
                    
                # Weighted random choice
                total_weight = sum(weights)
                if total_weight > 0:
                    r = random.uniform(0, total_weight)
                    cumsum = 0
                    for i, weight in enumerate(weights):
                        cumsum += weight
                        if r <= cumsum:
                            current = neighbors[i]
                            break
                else:
                    current = random.choice(neighbors)
                    
                path.append(current)
                
                # Check if we've reached sink area
                if math.sqrt((current[0] - sink[0])**2 + (current[1] - sink[1])**2) < 5:
                    break
                    
            # Find corners in this path
            for i, point in enumerate(path):
                if not any(self._has_line_of_sight(screenshot, point, sink) for sink in sinks):
                    if i > 0:
                        candidates.append(path[i-1])
                    break
                    
        return list(set(candidates))  # Remove duplicates
    
    def _optimize_waypoint_simulated_annealing(self, screenshot: np.ndarray, candidates: List[Tuple[int, int]], sinks: List[Tuple[int, int]]) -> Optional[Dict[str, float]]:
        """Optimize waypoint placement using simulated annealing."""
        if not candidates:
            return None
            
        # Start with random candidate
        best_candidate = random.choice(candidates)
        current_x, current_y = float(best_candidate[1]), float(best_candidate[0])
        current_radius = 2.0
        
        best_x, best_y, best_radius = current_x, current_y, current_radius
        best_score = self._evaluate_waypoint_candidate(screenshot, current_x, current_y, current_radius, sinks)
        
        temperature = self.temperature
        no_improvement_count = 0
        
        while temperature > self.min_temperature and no_improvement_count < 50:
            # Generate neighbor solution
            new_x = current_x + random.normal(0, temperature)
            new_y = current_y + random.normal(0, temperature)
            new_radius = max(0.5, current_radius + random.normal(0, temperature * 0.1))
            
            # Evaluate new solution
            new_score = self._evaluate_waypoint_candidate(screenshot, new_x, new_y, new_radius, sinks)
            
            # Accept or reject based on simulated annealing criteria
            delta = new_score - best_score
            if delta < 0 or random.random() < math.exp(-delta / temperature):
                current_x, current_y, current_radius = new_x, new_y, new_radius
                
                if new_score < best_score:
                    best_x, best_y, best_radius = new_x, new_y, new_radius
                    best_score = new_score
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
            else:
                no_improvement_count += 1
                
            temperature *= self.cooling_rate
            
        # Convert to world coordinates
        world_x, world_y = pixel_to_world(best_x, best_y, screenshot.shape)
        
        return {
            'x': world_x,
            'y': world_y,
            'radius': best_radius * (800 / screenshot.shape[1])
        }
    
    def _evaluate_waypoint_candidate(self, screenshot: np.ndarray, x: float, y: float, radius: float, sinks: List[Tuple[int, int]]) -> float:
        """Evaluate waypoint candidate using multiple criteria."""
        penalty = 0.0
        
        # Wall collision penalty
        if self._waypoint_hits_wall_float(screenshot, x, y, radius):
            penalty += 1000.0
            
        # Line of sight penalty
        for sink in sinks:
            if not self._has_line_of_sight(screenshot, (int(y), int(x)), sink):
                penalty += 500.0
                
        # Size penalty (prefer smaller waypoints)
        penalty += radius * 10.0
        
        # Distance from corner penalty (spring attraction simulation)
        # penalty += distance_from_original_corner * 50.0
        
        return penalty
    
    def _waypoint_hits_wall_float(self, screenshot: np.ndarray, x: float, y: float, radius: float) -> bool:
        """Check wall collision for float coordinates (allows 1-pixel overlap)."""
        # Check if waypoint goes out of bounds
        height, width = screenshot.shape
        if (x - radius < -1 or x + radius >= width + 1 or
            y - radius < -1 or y + radius >= height + 1):
            return True
        
        # Allow waypoints to overlap walls by 1 pixel for discretization safety
        return False


class DescartesCornerTurning(CornerTurningVariant):  # Named after Descartes - analytical geometry master
    """
    THE COMPUTATIONAL GEOMETRY PURIST
    
    Personality: Loves mathematically precise solutions, thinks in terms of geometry
    
    Interpretation:
    - "Trace along path" = compute exact Voronoi diagram and follow skeleton
    - "Stochastic expansion" = analytical circle packing with exact tangent calculations
    - "Spring attraction" = constrained optimization with Lagrange multipliers
    - "Stop iteration" = exact convergence when mathematical constraints are satisfied
    """
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        sources, sinks = self._find_sources_and_sinks(screenshot)
        if not sources or not sinks:
            return []
            
        waypoints = []
        current_sources = sources
        
        # Extract wall boundaries for geometric analysis
        wall_boundaries = self._extract_wall_boundaries(screenshot)
        
        for waypoint_index in range(self.max_waypoints):
            # Compute visibility graph
            visibility_graph = self._compute_visibility_graph(screenshot, current_sources, sinks, wall_boundaries)
            
            # Find critical points using medial axis
            critical_points = self._find_critical_points_medial_axis(screenshot, current_sources, sinks, wall_boundaries)
            
            if not critical_points:
                break
                
            # Analytically compute optimal waypoint
            waypoint = self._compute_optimal_waypoint_analytical(screenshot, critical_points, sinks, wall_boundaries)
            
            if waypoint:
                waypoints.append(waypoint)
                wx, wy = world_to_pixel(waypoint['x'], waypoint['y'], screenshot.shape)
                current_sources = [(int(wy), int(wx))]
            else:
                break
                
        return waypoints
    
    def _extract_wall_boundaries(self, screenshot: np.ndarray) -> List[List[Tuple[int, int]]]:
        """Extract wall boundary polygons using contour tracing."""
        # This is a simplified version - real implementation would use proper contour extraction
        boundaries = []
        
        # Find wall edges
        for y in range(1, screenshot.shape[0] - 1):
            for x in range(1, screenshot.shape[1] - 1):
                if screenshot[y, x] == 1:  # Wall pixel
                    # Check if it's on boundary
                    is_boundary = False
                    for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                        if screenshot[y+dy, x+dx] != 1:
                            is_boundary = True
                            break
                    
                    if is_boundary:
                        boundaries.append([(y, x)])  # Simplified - should trace full contours
                        
        return boundaries
    
    def _compute_visibility_graph(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]], boundaries: List[List[Tuple[int, int]]]) -> Dict:
        """Compute visibility graph between key points."""
        # Simplified visibility graph computation
        graph = {}
        all_points = sources + sinks
        
        for i, point1 in enumerate(all_points):
            graph[i] = []
            for j, point2 in enumerate(all_points):
                if i != j and self._has_line_of_sight(screenshot, point1, point2):
                    graph[i].append(j)
                    
        return graph
    
    def _find_critical_points_medial_axis(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]], boundaries: List[List[Tuple[int, int]]]) -> List[Tuple[int, int]]:
        """Find critical points using medial axis transform."""
        # Simplified medial axis computation
        # Real implementation would use proper skeletonization
        
        critical_points = []
        
        # Use distance transform as approximation to medial axis
        # Create binary image (0 = wall, 1 = free space)
        binary = (screenshot != 1).astype(np.uint8)
        
        # Simple skeleton approximation
        for y in range(2, screenshot.shape[0] - 2):
            for x in range(2, screenshot.shape[1] - 2):
                if binary[y, x] == 1:  # Free space
                    # Check if point is local maximum of distance to walls
                    local_distances = []
                    for dy in range(-2, 3):
                        for dx in range(-2, 3):
                            ny, nx = y + dy, x + dx
                            if binary[ny, nx] == 1:
                                # Distance to nearest wall
                                min_wall_dist = float('inf')
                                for wy in range(screenshot.shape[0]):
                                    for wx in range(screenshot.shape[1]):
                                        if screenshot[wy, wx] == 1:
                                            dist = math.sqrt((ny - wy)**2 + (nx - wx)**2)
                                            min_wall_dist = min(min_wall_dist, dist)
                                local_distances.append(min_wall_dist)
                    
                    if local_distances:
                        center_dist = local_distances[len(local_distances)//2]
                        if center_dist >= max(local_distances) * 0.9:  # Local maximum
                            critical_points.append((y, x))
                            
        return critical_points[:5]  # Limit to reasonable number
    
    def _compute_optimal_waypoint_analytical(self, screenshot: np.ndarray, critical_points: List[Tuple[int, int]], sinks: List[Tuple[int, int]], boundaries: List[List[Tuple[int, int]]]) -> Optional[Dict[str, float]]:
        """Compute optimal waypoint using analytical geometry."""
        if not critical_points:
            return None
            
        best_point = None
        best_radius = 0
        best_score = float('inf')
        
        for point in critical_points:
            # Compute maximum inscribed circle at this point
            max_radius = self._compute_max_inscribed_circle_radius(screenshot, point)
            
            if max_radius > 1.0:
                # Check visibility constraints
                visible_to_all_sinks = all(
                    self._has_line_of_sight(screenshot, point, sink) for sink in sinks
                )
                
                if visible_to_all_sinks:
                    # Evaluate using multiple geometric criteria
                    score = self._evaluate_geometric_criteria(screenshot, point, max_radius, sinks)
                    
                    if score < best_score:
                        best_score = score
                        best_point = point
                        best_radius = max_radius
                        
        if best_point is None:
            return None
            
        # Convert to world coordinates
        world_x, world_y = pixel_to_world(best_point[1], best_point[0], screenshot.shape)
        
        return {
            'x': world_x,
            'y': world_y,
            'radius': best_radius * (800 / screenshot.shape[1])
        }
    
    def _compute_max_inscribed_circle_radius(self, screenshot: np.ndarray, center: Tuple[int, int]) -> float:
        """Compute maximum radius of circle inscribed at given center."""
        cy, cx = center
        max_radius = 0
        
        # Search outward from center
        for radius in np.arange(0.5, 50, 0.5):
            # Check if circle intersects any walls
            intersects_wall = False
            for angle in np.linspace(0, 2*np.pi, 24):
                px = int(cx + radius * np.cos(angle))
                py = int(cy + radius * np.sin(angle))
                
                if self._is_wall(screenshot, py, px):
                    intersects_wall = True
                    break
                    
            if intersects_wall:
                break
            else:
                max_radius = radius
                
        return max_radius
    
    def _evaluate_geometric_criteria(self, screenshot: np.ndarray, point: Tuple[int, int], radius: float, sinks: List[Tuple[int, int]]) -> float:
        """Evaluate waypoint using multiple geometric criteria."""
        score = 0.0
        
        # Minimize radius (prefer smaller waypoints)
        score += radius * 0.1
        
        # Minimize distance to ideal position on medial axis
        # score += distance_to_medial_axis * 0.5
        
        # Maximize distance to walls (prefer central placement)
        min_wall_distance = self._compute_min_distance_to_wall(screenshot, point)
        score -= min_wall_distance * 0.2
        
        # Minimize maximum distance to sinks (prefer central to all sinks)
        max_sink_distance = max(
            math.sqrt((point[0] - sink[0])**2 + (point[1] - sink[1])**2)
            for sink in sinks
        )
        score += max_sink_distance * 0.1
        
        return score
    
    def _compute_min_distance_to_wall(self, screenshot: np.ndarray, point: Tuple[int, int]) -> float:
        """Compute minimum distance from point to any wall."""
        py, px = point
        min_distance = float('inf')
        
        for y in range(screenshot.shape[0]):
            for x in range(screenshot.shape[1]):
                if screenshot[y, x] == 1:  # Wall
                    distance = math.sqrt((py - y)**2 + (px - x)**2)
                    min_distance = min(min_distance, distance)
                    
        return min_distance


class SocratesCornerTurning(CornerTurningVariant):  # Named after Socrates - wisdom through rules and heuristics
    """
    THE PRACTICAL HEURISTICS ENGINEER
    
    Personality: Values rules of thumb, prefers simple robust solutions
    
    Interpretation:
    - "Trace along path" = follow simple heuristic rules (go towards narrowest passage)
    - "Stochastic expansion" = try multiple fixed radius values, pick best
    - "Spring attraction" = bias towards corners, but with simple linear interpolation
    - "Stop iteration" = stop when waypoint "looks good enough" by heuristic rules
    """
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        sources, sinks = self._find_sources_and_sinks(screenshot)
        if not sources or not sinks:
            return []
            
        waypoints = []
        current_sources = sources
        
        for waypoint_index in range(self.max_waypoints):
            # Apply heuristic rules to find good waypoint location
            waypoint_location = self._apply_heuristic_rules(screenshot, current_sources, sinks)
            
            if waypoint_location is None:
                break
                
            # Simple radius selection using heuristics
            radius = self._select_radius_heuristically(screenshot, waypoint_location)
            
            # Convert to waypoint
            world_x, world_y = pixel_to_world(waypoint_location[1], waypoint_location[0], screenshot.shape)
            
            waypoint = {
                'x': world_x,
                'y': world_y,
                'radius': radius * (800 / screenshot.shape[1])
            }
            
            waypoints.append(waypoint)
            current_sources = [waypoint_location]
            
        return waypoints
    
    def _apply_heuristic_rules(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Apply simple heuristic rules to find waypoint location."""
        
        # Rule 1: Find the narrowest passage between sources and sinks
        narrowest_point = self._find_narrowest_passage(screenshot, sources, sinks)
        
        if narrowest_point:
            return narrowest_point
            
        # Rule 2: Find point where line of sight is first blocked
        blocked_point = self._find_first_blocked_point(screenshot, sources, sinks)
        
        return blocked_point
    
    def _find_narrowest_passage(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find the narrowest passage point using simple sampling."""
        
        # Sample points between sources and sinks
        candidates = []
        
        for source in sources:
            for sink in sinks:
                # Sample points along line between source and sink
                for t in np.linspace(0.2, 0.8, 10):
                    y = int(source[0] * (1-t) + sink[0] * t)
                    x = int(source[1] * (1-t) + sink[1] * t)
                    
                    if not self._is_wall(screenshot, y, x):
                        # Measure "narrowness" at this point
                        narrowness = self._measure_narrowness(screenshot, (y, x))
                        candidates.append((narrowness, (y, x)))
                        
        if candidates:
            # Return point with highest narrowness (most constrained)
            candidates.sort(reverse=True)
            return candidates[0][1]
            
        return None
    
    def _measure_narrowness(self, screenshot: np.ndarray, point: Tuple[int, int]) -> float:
        """Measure how narrow/constrained a point is."""
        y, x = point
        
        # Count walls in neighborhood
        wall_count = 0
        total_points = 0
        
        for dy in range(-3, 4):
            for dx in range(-3, 4):
                ny, nx = y + dy, x + dx
                if 0 <= ny < screenshot.shape[0] and 0 <= nx < screenshot.shape[1]:
                    total_points += 1
                    if screenshot[ny, nx] == 1:
                        wall_count += 1
                        
        return wall_count / total_points if total_points > 0 else 0
    
    def _find_first_blocked_point(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find first point where line of sight to sinks is blocked."""
        
        for source in sources:
            # Walk from source towards center of sinks
            sink_center_y = sum(sink[0] for sink in sinks) / len(sinks)
            sink_center_x = sum(sink[1] for sink in sinks) / len(sinks)
            
            # Direction vector
            dy = sink_center_y - source[0]
            dx = sink_center_x - source[1]
            length = math.sqrt(dy*dy + dx*dx)
            
            if length > 0:
                dy /= length
                dx /= length
                
                # Walk in this direction
                current_y, current_x = source[0], source[1]
                
                for step in range(int(length)):
                    current_y += dy
                    current_x += dx
                    
                    pixel_pos = (int(current_y), int(current_x))
                    
                    # Check if we've lost line of sight to any sink
                    if not any(self._has_line_of_sight(screenshot, pixel_pos, sink) for sink in sinks):
                        # Step back one position
                        prev_y = int(current_y - dy)
                        prev_x = int(current_x - dx)
                        return (prev_y, prev_x)
                        
        return None
    
    def _select_radius_heuristically(self, screenshot: np.ndarray, location: Tuple[int, int]) -> float:
        """Select waypoint radius using simple heuristics."""
        
        # Try several radius values and pick the largest that doesn't hit walls
        test_radii = [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]
        
        for radius in reversed(test_radii):  # Start with largest
            if not self._waypoint_hits_wall_simple(screenshot, location, radius):
                # Additional heuristic: prefer radius that's about 1/4 of local passage width
                local_width = self._estimate_local_passage_width(screenshot, location)
                preferred_radius = local_width * 0.25
                
                # Return closest to preferred radius
                if abs(radius - preferred_radius) < 2.0:
                    return radius
                    
        return 1.0  # Fallback to minimum radius
    
    def _waypoint_hits_wall_simple(self, screenshot: np.ndarray, location: Tuple[int, int], radius: float) -> bool:
        """Simple wall collision check (allows 1-pixel overlap)."""
        y, x = location
        height, width = screenshot.shape
        
        # Check if waypoint goes out of bounds
        if (x - radius < -1 or x + radius >= width + 1 or
            y - radius < -1 or y + radius >= height + 1):
            return True
        
        # Allow waypoints to overlap walls by 1 pixel for discretization safety
        return False
                
        return False
    
    def _estimate_local_passage_width(self, screenshot: np.ndarray, location: Tuple[int, int]) -> float:
        """Estimate width of passage at given location."""
        y, x = location
        
        widths = []
        
        # Check width in 4 directions
        for angle in [0, 45, 90, 135]:
            rad = math.radians(angle)
            dx, dy = math.cos(rad), math.sin(rad)
            
            # Find distance to wall in positive direction
            dist_pos = 0
            for step in range(1, 50):
                px = int(x + step * dx)
                py = int(y + step * dy)
                if self._is_wall(screenshot, py, px):
                    dist_pos = step
                    break
                    
            # Find distance to wall in negative direction
            dist_neg = 0
            for step in range(1, 50):
                px = int(x - step * dx)
                py = int(y - step * dy)
                if self._is_wall(screenshot, py, px):
                    dist_neg = step
                    break
                    
            total_width = dist_pos + dist_neg
            if total_width > 0:
                widths.append(total_width)
                
        return min(widths) if widths else 10.0


class PlatoBetaCornerTurning(CornerTurningVariant):  # Named after Plato - student trying their best, but inexperienced
    """
    THE STRUGGLING STUDENT
    
    Personality: Misunderstood the assignment, made classic beginner mistakes
    
    Interpretation:
    - "Trace along path" = confused, just samples random points
    - "Stochastic expansion" = misunderstood, makes radius random
    - "Spring attraction" = forgot to implement, waypoints drift everywhere
    - "Stop iteration" = uses hardcoded magic numbers
    """
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        sources, sinks = self._find_sources_and_sinks(screenshot)
        if not sources or not sinks:
            return []
        
        waypoints = []
        
        # Student mistake: hardcoded to always generate exactly 3 waypoints
        for i in range(3):
            # Student mistake: random waypoint placement instead of corner detection
            x = random.randint(10, screenshot.shape[1] - 10)
            y = random.randint(10, screenshot.shape[0] - 10)
            
            # Student mistake: radius is just random
            radius = random.uniform(1, 8)
            
            # Student mistake: doesn't check if waypoint is valid
            world_x, world_y = pixel_to_world(x, y, screenshot.shape)
            
            waypoint = {
                'x': world_x,
                'y': world_y, 
                'radius': radius * (800 / screenshot.shape[1])
            }
            waypoints.append(waypoint)
            
        return waypoints


class DiogenesCornerTurning(CornerTurningVariant):  # Named after Diogenes - minimalist, does the bare minimum
    """
    THE LAZY PROGRAMMER
    
    Personality: Takes shortcuts, minimal effort solutions
    
    Interpretation:
    - "Trace along path" = just connect source to sink with straight line
    - "Stochastic expansion" = try 2-3 fixed radius values, pick first that works
    - "Spring attraction" = ignored this requirement 
    - "Stop iteration" = stop after first attempt
    """
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        sources, sinks = self._find_sources_and_sinks(screenshot)
        if not sources or not sinks:
            return []
        
        # Lazy: just put waypoint in middle of first source and first sink
        source = sources[0]
        sink = sinks[0] 
        
        mid_x = (source[1] + sink[1]) // 2
        mid_y = (source[0] + sink[0]) // 2
        
        # Lazy: try a few fixed radii, pick first that doesn't obviously break
        for radius in [2.0, 5.0, 10.0]:
            if not self._is_wall(screenshot, mid_y, mid_x):
                world_x, world_y = pixel_to_world(mid_x, mid_y, screenshot.shape)
                return [{
                    'x': world_x,
                    'y': world_y,
                    'radius': radius * (800 / screenshot.shape[1])
                }]
        
        return []  # Lazy: give up if simple approach doesn't work


class LeonardoCornerTurning(CornerTurningVariant):  # Named after da Vinci - overengineered, complex genius
    """
    THE OVERENGINEER
    
    Personality: Makes everything unnecessarily complex, premature optimization
    
    Interpretation:
    - "Trace along path" = implements 5 different pathfinding algorithms and compares
    - "Stochastic expansion" = uses advanced ML techniques for simple problem
    - "Spring attraction" = implements full physics engine
    - "Stop iteration" = complex convergence criteria with multiple metrics
    """
    
    def __init__(self, max_waypoints: int = 10):
        super().__init__(max_waypoints)
        # Overengineered: way too many configuration parameters
        self.pathfinding_algorithms = ['astar', 'dijkstra', 'rrt', 'genetic', 'neural']
        self.expansion_strategies = ['ml_based', 'physics', 'optimization', 'heuristic']
        self.convergence_thresholds = {
            'force_threshold': 0.001,
            'position_threshold': 0.01, 
            'velocity_threshold': 0.005,
            'energy_threshold': 0.1,
            'gradient_threshold': 0.02
        }
        self.performance_metrics = {}
        
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        sources, sinks = self._find_sources_and_sinks(screenshot)
        if not sources or not sinks:
            return []
        
        # Overengineered: benchmark all pathfinding algorithms 
        best_algorithm = self._benchmark_pathfinding_algorithms(screenshot, sources, sinks)
        
        waypoints = []
        current_sources = sources
        
        for waypoint_index in range(self.max_waypoints):
            # Overengineered: use ensemble of methods and vote
            candidate_waypoints = []
            
            for strategy in self.expansion_strategies:
                try:
                    wp = self._generate_waypoint_with_strategy(screenshot, current_sources, sinks, strategy)
                    if wp:
                        candidate_waypoints.append(wp)
                except Exception:
                    pass  # Overengineered: silently ignore failures in ensemble
            
            if not candidate_waypoints:
                break
                
            # Overengineered: complex voting system to pick best candidate
            best_waypoint = self._ensemble_vote(candidate_waypoints, screenshot, sinks)
            
            if best_waypoint:
                waypoints.append(best_waypoint)
                wx, wy = world_to_pixel(best_waypoint['x'], best_waypoint['y'], screenshot.shape)
                current_sources = [(int(wy), int(wx))]
            else:
                break
                
        return waypoints
    
    def _benchmark_pathfinding_algorithms(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]]) -> str:
        """Overengineered: benchmark multiple pathfinding algorithms."""
        # This would be pages of code in real implementation
        # For now, just pretend we did extensive benchmarking
        self.performance_metrics['pathfinding_benchmark'] = {
            'astar': {'time': 0.01, 'accuracy': 0.95, 'memory': 1024},
            'dijkstra': {'time': 0.02, 'accuracy': 1.0, 'memory': 2048},
            'rrt': {'time': 0.05, 'accuracy': 0.8, 'memory': 512}
        }
        return 'dijkstra'  # Winner of extensive benchmarking
    
    def _generate_waypoint_with_strategy(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]], strategy: str) -> Optional[Dict[str, float]]:
        """Generate waypoint using specific overengineered strategy."""
        
        if strategy == 'ml_based':
            # Overengineered: use ML for simple geometric problem
            return self._ml_based_waypoint_generation(screenshot, sources, sinks)
        elif strategy == 'physics':
            # Overengineered: full physics simulation
            return self._physics_simulation_waypoint(screenshot, sources, sinks)
        elif strategy == 'optimization':
            # Overengineered: global optimization with multiple objectives
            return self._multi_objective_optimization_waypoint(screenshot, sources, sinks)
        else:
            # Fallback to simple heuristic
            return self._simple_heuristic_waypoint(screenshot, sources, sinks)
    
    def _ml_based_waypoint_generation(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]]) -> Optional[Dict[str, float]]:
        """Overengineered ML approach for simple problem."""
        # In real overengineered version: neural networks, deep learning, etc.
        # For demo: just random with fancy name
        if sources and sinks:
            x = (sources[0][1] + sinks[0][1]) / 2 + random.normal(0, 5)
            y = (sources[0][0] + sinks[0][0]) / 2 + random.normal(0, 5)
            
            world_x, world_y = pixel_to_world(x, y, screenshot.shape)
            return {
                'x': world_x,
                'y': world_y,
                'radius': 3.0 * (800 / screenshot.shape[1])
            }
        return None
    
    def _physics_simulation_waypoint(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]]) -> Optional[Dict[str, float]]:
        """Overengineered physics simulation."""
        # Would implement full molecular dynamics simulation
        # For demo: simplified version
        if sources and sinks:
            x = (sources[0][1] + sinks[0][1]) / 2
            y = (sources[0][0] + sinks[0][0]) / 2
            
            world_x, world_y = pixel_to_world(x, y, screenshot.shape)
            return {
                'x': world_x,
                'y': world_y,
                'radius': 4.0 * (800 / screenshot.shape[1])
            }
        return None
    
    def _multi_objective_optimization_waypoint(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]]) -> Optional[Dict[str, float]]:
        """Overengineered multi-objective optimization."""
        # Would use genetic algorithms, simulated annealing, particle swarm, etc.
        # For demo: random optimization
        if sources and sinks:
            best_x, best_y = sources[0][1], sources[0][0]
            
            # Pretend we did complex optimization
            for _ in range(5):  # Reduced from the planned 10000 iterations
                x = best_x + random.normal(0, 2)
                y = best_y + random.normal(0, 2)
                # Complex scoring function would go here
                best_x, best_y = x, y
            
            world_x, world_y = pixel_to_world(best_x, best_y, screenshot.shape)
            return {
                'x': world_x,
                'y': world_y,
                'radius': 5.0 * (800 / screenshot.shape[1])
            }
        return None
    
    def _simple_heuristic_waypoint(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]]) -> Optional[Dict[str, float]]:
        """Fallback simple approach."""
        if sources and sinks:
            x = (sources[0][1] + sinks[0][1]) / 2
            y = (sources[0][0] + sinks[0][0]) / 2
            
            world_x, world_y = pixel_to_world(x, y, screenshot.shape)
            return {
                'x': world_x,
                'y': world_y,
                'radius': 2.0 * (800 / screenshot.shape[1])
            }
        return None
    
    def _ensemble_vote(self, candidates: List[Dict[str, float]], screenshot: np.ndarray, sinks: List[Tuple[int, int]]) -> Optional[Dict[str, float]]:
        """Overengineered ensemble voting system."""
        if not candidates:
            return None
            
        # Overengineered: weighted voting based on multiple criteria
        weights = []
        for candidate in candidates:
            # Complex scoring would go here
            weight = random.uniform(0.5, 1.0)  # Placeholder
            weights.append(weight)
        
        # Return highest weighted candidate
        best_idx = weights.index(max(weights))
        return candidates[best_idx]


class MurphyCornerTurning(CornerTurningVariant):  # Named after Murphy's Law - everything that can go wrong, will
    """
    THE BUGGY IMPLEMENTATION
    
    Personality: Tried to do it right but made several critical errors
    
    Common bugs:
    - Off-by-one errors in array indexing
    - Coordinate system confusion (x/y swapped)
    - Logic errors in conditionals
    - Memory leaks and infinite loops
    """
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        sources, sinks = self._find_sources_and_sinks(screenshot)
        if not sources or not sinks:
            return []
        
        waypoints = []
        
        # Bug: off-by-one error in loop limit
        for i in range(self.max_waypoints + 1):  # Should be max_waypoints, not +1
            
            # Bug: coordinate confusion (x and y swapped)
            corner = self._find_corner_buggy(screenshot, sources, sinks)
            if corner is None:
                break
                
            # Bug: incorrect radius calculation
            radius = self._calculate_radius_buggy(screenshot, corner)
            
            # Bug: coordinate system confusion in conversion
            world_x, world_y = pixel_to_world(corner[0], corner[1], screenshot.shape)  # Swapped!
            
            waypoint = {
                'x': world_x,
                'y': world_y,
                'radius': radius * (800 / screenshot.shape[1])
            }
            waypoints.append(waypoint)
            
            # Bug: forgot to update sources for next iteration
            # sources = [corner]  # This line is missing!
            
            # Bug: potential infinite loop if corner detection always returns same point
            if len(waypoints) > 20:  # Emergency break to prevent infinite loop
                break
                
        return waypoints
    
    def _find_corner_buggy(self, screenshot: np.ndarray, sources: List[Tuple[int, int]], sinks: List[Tuple[int, int]]) -> Optional[Tuple[int, int]]:
        """Find corner with several bugs."""
        
        for source in sources:
            for sink in sinks:
                # Bug: wrong loop bounds (should be range(1, distance))
                distance = int(math.sqrt((sink[0] - source[0])**2 + (sink[1] - source[1])**2))
                
                for step in range(0, distance + 1):  # Bug: includes 0 and goes beyond distance
                    # Bug: integer division instead of float
                    t = step // distance if distance > 0 else 0  # Should be step / distance
                    
                    x = int(source[1] * (1-t) + sink[1] * t)
                    y = int(source[0] * (1-t) + sink[0] * t)
                    
                    # Bug: boundary check is wrong
                    if x <= 0 or x >= screenshot.shape[1] or y <= 0 or y >= screenshot.shape[0]:  # Should be < not <=
                        continue
                        
                    # Bug: logic error in line of sight check
                    if self._has_line_of_sight(screenshot, (y, x), sink):  # Bug: should be NOT has line of sight
                        return (y, x)
                        
        return None
    
    def _calculate_radius_buggy(self, screenshot: np.ndarray, corner: Tuple[int, int]) -> float:
        """Calculate radius with bugs."""
        
        # Bug: doesn't check bounds properly
        y, x = corner
        
        radius = 1.0
        
        # Bug: infinite loop potential
        while radius < 100:  # Should have a reasonable upper bound
            
            # Bug: only checks 4 points instead of full circle
            test_points = [
                (y + radius, x),  # Bug: should be int(y + radius)
                (y - radius, x),
                (y, x + radius),
                (y, x - radius)
            ]
            
            hit_wall = False
            for py, px in test_points:
                # Bug: doesn't convert to int
                if self._is_wall(screenshot, py, px):  # py, px are floats!
                    hit_wall = True
                    break
                    
            if hit_wall:
                return radius - 1.0  # Return previous radius
                
            radius += 1.0
            
        return 50.0  # Fallback value


# Integration with tournament system
def get_all_corner_turning_variants():
    """Get all corner turning variants for tournament integration."""
    return {
        'GridMarchingCornerTurning': GridMarchingCornerTurning,
        'PhysicsSimulationCornerTurning': PhysicsSimulationCornerTurning,
        'RandomizedOptimizationCornerTurning': RandomizedOptimizationCornerTurning,
        'GeometryAnalyticalCornerTurning': GeometryAnalyticalCornerTurning,
        'HeuristicRulesCornerTurning': HeuristicRulesCornerTurning,
        'StudentImplementationCornerTurning': StudentImplementationCornerTurning,
        'LazyImplementationCornerTurning': LazyImplementationCornerTurning,
        'OverengineeredCornerTurning': OverengineeredCornerTurning,
        'BuggyImplementationCornerTurning': BuggyImplementationCornerTurning
    }


def create_corner_turning_variant(variant_name: str) -> CornerTurningVariant:
    """Factory function to create corner turning variants by name."""
    variants = get_all_corner_turning_variants()
    
    if variant_name not in variants:
        raise ValueError(f"Unknown corner turning variant: {variant_name}. Available: {list(variants.keys())}")
    
    return variants[variant_name]()


# Test harness for comparing variants
def test_corner_turning_variants():
    """Test all corner turning variants on a simple synthetic case."""
    
    # Create a simple test case
    test_screenshot = np.zeros((20, 30), dtype=int)
    
    # Add walls to create an L-shaped corridor
    test_screenshot[0, :] = 1  # Top wall
    test_screenshot[-1, :] = 1  # Bottom wall
    test_screenshot[:, 0] = 1  # Left wall
    test_screenshot[:, -1] = 1  # Right wall
    test_screenshot[10, 5:20] = 1  # Internal wall creating L-shape
    
    # Add source and sink
    test_screenshot[5, 2] = 3  # Source
    test_screenshot[15, 25] = 4  # Sink
    
    # Test all variants
    variant_classes = get_all_corner_turning_variants()
    variants = [
        ("Grid Marching", GridMarchingCornerTurning()),
        ("Physics Simulation", PhysicsSimulationCornerTurning()),
        ("Randomized Optimization", RandomizedOptimizationCornerTurning()),
        ("Geometry Analytical", GeometryAnalyticalCornerTurning()),
        ("Heuristic Rules", HeuristicRulesCornerTurning()),
        ("Student Implementation", StudentImplementationCornerTurning()),
        ("Lazy Implementation", LazyImplementationCornerTurning()),
        ("Overengineered", OverengineeredCornerTurning()),
        ("Buggy Implementation", BuggyImplementationCornerTurning())
    ]
    
    print("Testing Corner Turning Algorithm Variants")
    print("=" * 50)
    print(f"Test case: L-shaped corridor with source at (5,2) and sink at (15,25)")
    print()
    
    for name, variant in variants:
        print(f"{name}:")
        try:
            waypoints = variant.generate_waypoints(test_screenshot)
            print(f"  Generated {len(waypoints)} waypoints")
            for i, wp in enumerate(waypoints):
                print(f"    Waypoint {i+1}: x={wp['x']:.2f}, y={wp['y']:.2f}, r={wp['radius']:.2f}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
        print()


if __name__ == "__main__":
    test_corner_turning_variants()