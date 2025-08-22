"""
Waypoint generators inspired by 2024 robotics research.

Based on web search findings about modern path planning algorithms including:
- CNN-based waypoint optimization
- Sampling-based algorithms  
- Multi-algorithm fusion approaches
- Bio-inspired intelligent algorithms
- Narrow space navigation techniques
"""

import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
import heapq

try:
    from .waypoint_generation import WaypointGenerator
    from .improved_waypoint_scoring import improved_score_waypoint_list
except ImportError:
    from waypoint_generation import WaypointGenerator
    from improved_waypoint_scoring import improved_score_waypoint_list


class AStarWaypointGenerator(WaypointGenerator):
    """
    Uses A* pathfinding to identify critical waypoints along optimal path.
    
    Inspired by graph-based search algorithms mentioned in robotics research.
    Places waypoints at path decision points and narrow passages.
    """
    
    def __init__(self):
        super().__init__("AStarWaypoints")
    
    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Manhattan distance heuristic."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _find_path_astar(self, screenshot: np.ndarray, start: Tuple[int, int], 
                        goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Find optimal path using A* algorithm."""
        height, width = screenshot.shape
        
        # A* data structures
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                return path[::-1]
            
            for dx, dy in directions:
                neighbor = (current[0] + dx, current[1] + dy)
                
                if (neighbor[0] < 0 or neighbor[0] >= width or 
                    neighbor[1] < 0 or neighbor[1] >= height):
                    continue
                
                if screenshot[neighbor[1], neighbor[0]] == 1:  # Wall
                    continue
                
                tentative_g = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    
                    if neighbor not in [item[1] for item in open_set]:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
        
        return []
    
    def _identify_critical_points(self, path: List[Tuple[int, int]], 
                                 screenshot: np.ndarray) -> List[Tuple[int, int]]:
        """Identify critical decision points along path."""
        if len(path) < 3:
            return []
        
        critical_points = []
        height, width = screenshot.shape
        
        for i in range(1, len(path) - 1):
            x, y = path[i]
            
            # Count passable neighbors
            passable_neighbors = 0
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if (0 <= nx < width and 0 <= ny < height and 
                        screenshot[ny, nx] != 1):
                        passable_neighbors += 1
            
            # Critical if narrow passage or decision point
            if passable_neighbors <= 3 or self._is_direction_change(path, i):
                critical_points.append((x, y))
        
        return critical_points
    
    def _is_direction_change(self, path: List[Tuple[int, int]], index: int) -> bool:
        """Check if path changes direction significantly at this point."""
        if index <= 0 or index >= len(path) - 1:
            return False
        
        prev_point = path[index - 1]
        curr_point = path[index]
        next_point = path[index + 1]
        
        # Calculate direction vectors
        dir1 = (curr_point[0] - prev_point[0], curr_point[1] - prev_point[1])
        dir2 = (next_point[0] - curr_point[0], next_point[1] - curr_point[1])
        
        # Check if direction changed significantly
        return dir1 != dir2
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using A* path analysis."""
        sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
        sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
        
        if not sources or not sinks:
            return []
        
        waypoints = []
        
        # Find paths from each source to nearest sink
        for source in sources:
            # Find nearest sink
            nearest_sink = min(sinks, key=lambda s: abs(s[0] - source[0]) + abs(s[1] - source[1]))
            
            # Find optimal path
            path = self._find_path_astar(screenshot, source, nearest_sink)
            
            if path:
                # Identify critical waypoints along path
                critical_points = self._identify_critical_points(path, screenshot)
                
                for cp in critical_points[:3]:  # Limit waypoints
                    waypoints.append({
                        'x': float(cp[0]),
                        'y': float(cp[1]),
                        'radius': 6.0
                    })
        
        # Remove duplicates
        unique_waypoints = []
        for wp in waypoints:
            is_duplicate = False
            for existing in unique_waypoints:
                dist = math.sqrt((wp['x'] - existing['x'])**2 + (wp['y'] - existing['y'])**2)
                if dist < 10:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_waypoints.append(wp)
        
        return unique_waypoints[:4]  # Limit total waypoints


class SamplingBasedGenerator(WaypointGenerator):
    """
    Rapidly-exploring Random Tree (RRT) inspired waypoint generation.
    
    Based on sampling-based algorithms from robotics research that create
    searchable trees by randomly sampling nodes in state space.
    """
    
    def __init__(self, max_samples: int = 200):
        super().__init__("SamplingBased")
        self.max_samples = max_samples
    
    def _sample_free_space(self, screenshot: np.ndarray) -> Tuple[int, int]:
        """Sample random point in free space."""
        height, width = screenshot.shape
        
        for _ in range(100):  # Try up to 100 times
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            
            if screenshot[y, x] != 1:  # Not a wall
                return (x, y)
        
        # Fallback to center
        return (width // 2, height // 2)
    
    def _nearest_node(self, tree: List[Tuple[int, int]], 
                     sample: Tuple[int, int]) -> Tuple[int, int]:
        """Find nearest node in tree to sample point."""
        if not tree:
            return sample
        
        return min(tree, key=lambda node: 
                  (node[0] - sample[0])**2 + (node[1] - sample[1])**2)
    
    def _can_connect(self, screenshot: np.ndarray, start: Tuple[int, int], 
                    end: Tuple[int, int]) -> bool:
        """Check if two points can be connected without hitting walls."""
        x1, y1 = start
        x2, y2 = end
        
        # Bresenham's line algorithm
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        sx = 1 if x1 < x2 else -1
        sy = 1 if y1 < y2 else -1
        err = dx - dy
        
        x, y = x1, y1
        
        while True:
            if screenshot[y, x] == 1:  # Hit wall
                return False
            
            if x == x2 and y == y2:
                break
            
            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy
        
        return True
    
    def _build_rrt(self, screenshot: np.ndarray, start: Tuple[int, int], 
                  goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Build RRT from start toward goal."""
        height, width = screenshot.shape
        tree = [start]
        parent = {start: None}
        
        for _ in range(self.max_samples):
            # Bias sampling toward goal occasionally
            if random.random() < 0.1:
                sample = goal
            else:
                sample = self._sample_free_space(screenshot)
            
            nearest = self._nearest_node(tree, sample)
            
            # Extend toward sample
            direction_x = sample[0] - nearest[0]
            direction_y = sample[1] - nearest[1]
            distance = math.sqrt(direction_x**2 + direction_y**2)
            
            if distance < 1:
                continue
            
            # Step size
            step_size = min(10, distance)
            new_x = nearest[0] + int((direction_x / distance) * step_size)
            new_y = nearest[1] + int((direction_y / distance) * step_size)
            
            # Keep in bounds
            new_x = max(0, min(width - 1, new_x))
            new_y = max(0, min(height - 1, new_y))
            new_point = (new_x, new_y)
            
            # Check if connection is valid
            if self._can_connect(screenshot, nearest, new_point):
                tree.append(new_point)
                parent[new_point] = nearest
                
                # Check if goal is reachable
                goal_distance = math.sqrt((new_x - goal[0])**2 + (new_y - goal[1])**2)
                if goal_distance < 15 and self._can_connect(screenshot, new_point, goal):
                    tree.append(goal)
                    parent[goal] = new_point
                    break
        
        return tree
    
    def _extract_waypoints_from_tree(self, tree: List[Tuple[int, int]], 
                                   start: Tuple[int, int], 
                                   goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Extract strategic waypoints from RRT."""
        if goal not in tree:
            return []
        
        # Find branch points (nodes with multiple children)
        children_count = {}
        for node in tree:
            children_count[node] = 0
        
        # This is simplified - in full RRT we'd track parent relationships
        # For now, find nodes that are potential decision points
        waypoint_candidates = []
        
        for node in tree:
            if node == start or node == goal:
                continue
            
            # Check if node is at intersection or narrow passage
            x, y = node
            nearby_tree_nodes = [n for n in tree 
                               if math.sqrt((n[0]-x)**2 + (n[1]-y)**2) < 15]
            
            if len(nearby_tree_nodes) >= 3:  # Intersection point
                waypoint_candidates.append(node)
        
        # Select best candidates
        waypoint_candidates.sort(key=lambda p: 
                               min(math.sqrt((p[0]-start[0])**2 + (p[1]-start[1])**2),
                                   math.sqrt((p[0]-goal[0])**2 + (p[1]-goal[1])**2)))
        
        return waypoint_candidates[:3]
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using sampling-based tree exploration."""
        sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
        sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
        
        if not sources or not sinks:
            return []
        
        # Build RRT from first source to first sink
        source = sources[0]
        sink = sinks[0]
        
        tree = self._build_rrt(screenshot, source, sink)
        waypoint_positions = self._extract_waypoints_from_tree(tree, source, sink)
        
        waypoints = []
        for pos in waypoint_positions:
            waypoints.append({
                'x': float(pos[0]),
                'y': float(pos[1]),
                'radius': 7.0
            })
        
        return waypoints


class MultiAlgorithmFusionGenerator(WaypointGenerator):
    """
    Combines multiple algorithms as suggested by 2024 research showing
    "multi-algorithm fusion for path planning outperforms individual algorithms".
    
    Fuses results from different approaches to create robust waypoint sets.
    """
    
    def __init__(self):
        super().__init__("MultiAlgorithmFusion")
        # Initialize sub-algorithms
        self.astar_gen = AStarWaypointGenerator()
        self.sampling_gen = SamplingBasedGenerator(max_samples=100)
    
    def _merge_waypoint_sets(self, *waypoint_sets: List[List[Dict[str, float]]]) -> List[Dict[str, float]]:
        """Merge waypoints from multiple algorithms, removing duplicates and conflicts."""
        all_waypoints = []
        
        # Collect all waypoints
        for wp_set in waypoint_sets:
            all_waypoints.extend(wp_set)
        
        if not all_waypoints:
            return []
        
        # Remove duplicates and merge nearby waypoints
        merged_waypoints = []
        
        for wp in all_waypoints:
            # Check if similar waypoint already exists
            merged = False
            for i, existing in enumerate(merged_waypoints):
                dist = math.sqrt((wp['x'] - existing['x'])**2 + (wp['y'] - existing['y'])**2)
                
                if dist < 15:  # Merge nearby waypoints
                    # Take weighted average position and max radius
                    merged_waypoints[i] = {
                        'x': (wp['x'] + existing['x']) / 2,
                        'y': (wp['y'] + existing['y']) / 2,
                        'radius': max(wp['radius'], existing['radius'])
                    }
                    merged = True
                    break
            
            if not merged:
                merged_waypoints.append(wp.copy())
        
        return merged_waypoints
    
    def _rank_waypoints_by_importance(self, waypoints: List[Dict[str, float]], 
                                    screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Rank waypoints by strategic importance."""
        if not waypoints:
            return []
        
        scored_waypoints = []
        
        for wp in waypoints:
            importance_score = 0
            x, y = int(wp['x']), int(wp['y'])
            
            # Score based on local environment
            height, width = screenshot.shape
            
            # Check nearby wall density (higher = more important for navigation)
            wall_count = 0
            total_count = 0
            
            for dx in range(-10, 11):
                for dy in range(-10, 11):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < width and 0 <= ny < height:
                        total_count += 1
                        if screenshot[ny, nx] == 1:
                            wall_count += 1
            
            if total_count > 0:
                wall_density = wall_count / total_count
                importance_score += wall_density * 100  # Prefer areas with moderate wall density
            
            # Prefer waypoints between sources and sinks
            sources = [(sx, sy) for sy, sx in zip(*np.where(screenshot == 3))]
            sinks = [(sx, sy) for sy, sx in zip(*np.where(screenshot == 4))]
            
            if sources and sinks:
                min_source_dist = min(math.sqrt((x-sx)**2 + (y-sy)**2) for sx, sy in sources)
                min_sink_dist = min(math.sqrt((x-tx)**2 + (y-ty)**2) for tx, ty in sinks)
                
                # Prefer waypoints that are roughly equidistant from sources and sinks
                balance_score = 100 - abs(min_source_dist - min_sink_dist)
                importance_score += max(0, balance_score)
            
            scored_waypoints.append((wp, importance_score))
        
        # Sort by importance score (descending)
        scored_waypoints.sort(key=lambda x: x[1], reverse=True)
        
        return [wp for wp, score in scored_waypoints]
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using multi-algorithm fusion."""
        # Generate waypoints using different algorithms
        astar_waypoints = self.astar_gen.generate_waypoints(screenshot)
        sampling_waypoints = self.sampling_gen.generate_waypoints(screenshot)
        
        # Simple corner-based fallback
        corner_waypoints = self._generate_corner_waypoints(screenshot)
        
        # Merge results
        merged_waypoints = self._merge_waypoint_sets(
            astar_waypoints, sampling_waypoints, corner_waypoints
        )
        
        # Rank by importance
        ranked_waypoints = self._rank_waypoints_by_importance(merged_waypoints, screenshot)
        
        # Return top waypoints
        return ranked_waypoints[:4]
    
    def _generate_corner_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Simple corner detection as fallback."""
        height, width = screenshot.shape
        sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
        sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
        
        if not sources or not sinks:
            return []
        
        # Place waypoint between source and sink
        source = sources[0]
        sink = sinks[0]
        
        mid_x = (source[0] + sink[0]) / 2
        mid_y = (source[1] + sink[1]) / 2
        
        # Adjust if in wall
        for offset in range(20):
            for dx, dy in [(0,0), (offset,0), (-offset,0), (0,offset), (0,-offset)]:
                test_x = int(mid_x + dx)
                test_y = int(mid_y + dy)
                
                if (0 <= test_x < width and 0 <= test_y < height and 
                    screenshot[test_y, test_x] != 1):
                    return [{'x': float(test_x), 'y': float(test_y), 'radius': 8.0}]
        
        return []


class NarrowSpaceNavigationGenerator(WaypointGenerator):
    """
    Inspired by 2024 research on "Automatic Waypoint Generation to Improve 
    Robot Navigation Through Narrow Spaces".
    
    Specifically targets narrow passages and chokepoints for waypoint placement.
    """
    
    def __init__(self):
        super().__init__("NarrowSpaceNavigation")
    
    def _find_narrow_passages(self, screenshot: np.ndarray) -> List[Tuple[int, int, float]]:
        """Identify narrow passages in the level."""
        height, width = screenshot.shape
        narrow_passages = []
        
        # Analyze each passable cell for narrowness
        for y in range(2, height - 2):
            for x in range(2, width - 2):
                if screenshot[y, x] != 1:  # Passable cell
                    narrowness = self._calculate_narrowness(screenshot, x, y)
                    
                    if narrowness > 0.7:  # Threshold for narrow passages
                        narrow_passages.append((x, y, narrowness))
        
        # Sort by narrowness (higher = more narrow)
        narrow_passages.sort(key=lambda p: p[2], reverse=True)
        
        return narrow_passages
    
    def _calculate_narrowness(self, screenshot: np.ndarray, x: int, y: int) -> float:
        """Calculate how narrow the passage is at this point."""
        height, width = screenshot.shape
        
        # Check distances to walls in cardinal directions
        distances = []
        
        directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        
        for dx, dy in directions:
            distance = 0
            nx, ny = x, y
            
            while 0 <= nx < width and 0 <= ny < height and screenshot[ny, nx] != 1:
                distance += 1
                nx += dx
                ny += dy
                
                if distance > 20:  # Cap search distance
                    break
            
            distances.append(distance)
        
        if not distances:
            return 0
        
        # Narrowness is inverse of minimum distance to walls
        min_distance = min(distances)
        max_distance = max(distances)
        
        # High narrowness if close to walls on multiple sides
        if min_distance < 5:
            narrowness = 1.0 - (min_distance / 5.0)
            
            # Bonus for being narrow in multiple directions
            narrow_directions = sum(1 for d in distances if d < 8)
            narrowness += narrow_directions * 0.2
            
            return min(1.0, narrowness)
        
        return 0
    
    def _is_critical_passage(self, screenshot: np.ndarray, x: int, y: int) -> bool:
        """Check if this narrow passage is critical for connectivity."""
        height, width = screenshot.shape
        
        # Create temporary blocked version
        test_screenshot = screenshot.copy()
        
        # Block this position and small area around it
        for dx in range(-2, 3):
            for dy in range(-2, 3):
                nx, ny = x + dx, y + dy
                if 0 <= nx < width and 0 <= ny < height:
                    test_screenshot[ny, nx] = 1
        
        # Check if sources can still reach sinks
        sources = [(sx, sy) for sy, sx in zip(*np.where(screenshot == 3))]
        sinks = [(sx, sy) for sy, sx in zip(*np.where(screenshot == 4))]
        
        if not sources or not sinks:
            return False
        
        # Simple connectivity check
        from .level_validation import is_path_exists
        
        for source in sources:
            can_reach_any_sink = False
            for sink in sinks:
                if is_path_exists(test_screenshot, source, sink):
                    can_reach_any_sink = True
                    break
            
            if not can_reach_any_sink:
                return True  # Critical passage
        
        return False
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints focused on narrow space navigation."""
        narrow_passages = self._find_narrow_passages(screenshot)
        
        if not narrow_passages:
            return []
        
        waypoints = []
        
        # Select critical narrow passages
        for x, y, narrowness in narrow_passages[:10]:  # Check top 10 candidates
            if self._is_critical_passage(screenshot, x, y):
                # Calculate appropriate radius for narrow passage
                radius = max(3.0, min(8.0, 5.0 / narrowness))
                
                waypoints.append({
                    'x': float(x),
                    'y': float(y),
                    'radius': radius
                })
                
                if len(waypoints) >= 3:  # Limit waypoints
                    break
        
        return waypoints


def create_web_inspired_tournament():
    """Create tournament with web-inspired algorithms."""
    try:
        from .waypoint_generation import WaypointTournament
    except ImportError:
        from waypoint_generation import WaypointTournament
    
    tournament = WaypointTournament()
    
    # Add web-inspired algorithms
    tournament.add_generator(AStarWaypointGenerator())
    tournament.add_generator(SamplingBasedGenerator())
    tournament.add_generator(MultiAlgorithmFusionGenerator())
    tournament.add_generator(NarrowSpaceNavigationGenerator())
    
    return tournament


if __name__ == "__main__":
    # Test web-inspired algorithms
    print("Testing web-inspired waypoint algorithms...")
    
    from waypoint_test_runner import create_synthetic_test_cases
    
    test_cases = create_synthetic_test_cases()
    tournament = create_web_inspired_tournament()
    
    print(f"Created web-inspired tournament with {len(tournament.generators)} algorithms")
    
    # Test on all cases
    for test_name, screenshot in test_cases:
        print(f"\nTesting on {test_name}:")
        
        for generator in tournament.generators:
            try:
                import time
                start_time = time.time()
                waypoints = generator.generate_waypoints(screenshot)
                duration = time.time() - start_time
                
                score = improved_score_waypoint_list(screenshot, waypoints, 
                                                   penalize_skippable=True)
                
                print(f"  {generator.name:25} | {len(waypoints):2d} waypoints | {duration:5.3f}s | Score: {score:8.2f}")
                
            except Exception as e:
                print(f"  {generator.name:25} | ERROR: {e}")
    
    print("\nWeb-inspired algorithms test completed!")