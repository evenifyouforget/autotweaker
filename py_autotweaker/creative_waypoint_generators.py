"""
Creative waypoint generation algorithms inspired by various optimization and AI techniques.

This module implements novel approaches to waypoint generation including:
- Genetic algorithm optimization
- Particle swarm optimization  
- Reinforcement learning inspired approaches
- Game theory inspired approaches
- Fractal and recursive patterns
"""

import numpy as np
import math
import random
from typing import List, Dict, Tuple, Optional, Set
from collections import deque
import time

try:
    from .waypoint_generation import WaypointGenerator
    from .improved_waypoint_scoring import improved_score_waypoint_list
except ImportError:
    from waypoint_generation import WaypointGenerator
    from improved_waypoint_scoring import improved_score_waypoint_list


class GeneticWaypointGenerator(WaypointGenerator):
    """
    Generate waypoints using genetic algorithm optimization.
    
    Evolves populations of waypoint configurations over multiple generations.
    """
    
    def __init__(self, population_size: int = 20, generations: int = 50, 
                 mutation_rate: float = 0.3, max_waypoints: int = 8):
        super().__init__("Genetic")
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.max_waypoints = max_waypoints
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using genetic algorithm."""
        height, width = screenshot.shape
        
        # Find valid placement area
        passable_points = []
        for y in range(5, height - 5, 2):
            for x in range(5, width - 5, 2):
                if screenshot[y, x] != 1:
                    passable_points.append((x, y))
        
        if len(passable_points) < 10:
            return []
        
        # Initialize population
        population = []
        for _ in range(self.population_size):
            individual = self._create_random_individual(passable_points)
            population.append(individual)
        
        # Evolve population
        for generation in range(self.generations):
            # Evaluate fitness
            fitness_scores = []
            for individual in population:
                score = improved_score_waypoint_list(screenshot, individual, penalize_skippable=True)
                fitness_scores.append(1.0 / (1.0 + score))  # Convert to fitness (higher = better)
            
            # Selection and reproduction
            new_population = []
            
            # Keep best individuals (elitism)
            elite_count = max(2, self.population_size // 10)
            elite_indices = sorted(range(len(fitness_scores)), 
                                 key=lambda i: fitness_scores[i], reverse=True)[:elite_count]
            for i in elite_indices:
                new_population.append(population[i].copy())
            
            # Generate rest of population through crossover and mutation
            while len(new_population) < self.population_size:
                parent1 = self._tournament_selection(population, fitness_scores)
                parent2 = self._tournament_selection(population, fitness_scores)
                child = self._crossover(parent1, parent2, passable_points)
                child = self._mutate(child, passable_points)
                new_population.append(child)
            
            population = new_population
        
        # Return best individual
        final_scores = [improved_score_waypoint_list(screenshot, ind, penalize_skippable=True) 
                       for ind in population]
        best_idx = min(range(len(final_scores)), key=lambda i: final_scores[i])
        return population[best_idx]
    
    def _create_random_individual(self, passable_points: List[Tuple[int, int]]) -> List[Dict[str, float]]:
        """Create a random waypoint configuration."""
        num_waypoints = random.randint(0, self.max_waypoints)
        waypoints = []
        
        if num_waypoints > 0:
            selected_points = random.sample(passable_points, min(num_waypoints, len(passable_points)))
            for x, y in selected_points:
                radius = random.uniform(3.0, 15.0)
                waypoints.append({'x': float(x), 'y': float(y), 'radius': radius})
        
        return waypoints
    
    def _tournament_selection(self, population: List, fitness_scores: List[float]) -> List[Dict[str, float]]:
        """Select parent using tournament selection."""
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)
        best_idx = max(tournament_indices, key=lambda i: fitness_scores[i])
        return population[best_idx].copy()
    
    def _crossover(self, parent1: List, parent2: List, passable_points: List) -> List[Dict[str, float]]:
        """Create offspring through crossover."""
        # Combine waypoints from both parents
        all_waypoints = parent1 + parent2
        
        if len(all_waypoints) == 0:
            return self._create_random_individual(passable_points)
        
        # Select subset
        num_waypoints = min(random.randint(0, self.max_waypoints), len(all_waypoints))
        if num_waypoints == 0:
            return []
        
        selected = random.sample(all_waypoints, num_waypoints)
        return selected
    
    def _mutate(self, individual: List, passable_points: List) -> List[Dict[str, float]]:
        """Mutate an individual."""
        if random.random() > self.mutation_rate:
            return individual
        
        mutated = individual.copy()
        
        mutation_type = random.choice(['add', 'remove', 'modify'])
        
        if mutation_type == 'add' and len(mutated) < self.max_waypoints and passable_points:
            x, y = random.choice(passable_points)
            radius = random.uniform(3.0, 15.0)
            mutated.append({'x': float(x), 'y': float(y), 'radius': radius})
        
        elif mutation_type == 'remove' and len(mutated) > 0:
            mutated.pop(random.randint(0, len(mutated) - 1))
        
        elif mutation_type == 'modify' and len(mutated) > 0:
            idx = random.randint(0, len(mutated) - 1)
            waypoint = mutated[idx]
            if random.random() < 0.5:  # Modify position
                waypoint['x'] += random.uniform(-10, 10)
                waypoint['y'] += random.uniform(-10, 10)
            else:  # Modify radius
                waypoint['radius'] = max(2.0, waypoint['radius'] + random.uniform(-5, 5))
        
        return mutated


class FlowFieldGenerator(WaypointGenerator):
    """
    Generate waypoints by analyzing flow fields and potential functions.
    
    Creates a potential field from sources to sinks and places waypoints
    at critical points in the flow.
    """
    
    def __init__(self, max_waypoints: int = 6):
        super().__init__("FlowField")
        self.max_waypoints = max_waypoints
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using flow field analysis."""
        height, width = screenshot.shape
        
        # Create potential field
        potential_field = self._create_potential_field(screenshot)
        
        # Find critical points (local maxima/minima/saddle points)
        critical_points = self._find_critical_points(potential_field, screenshot)
        
        # Convert to waypoints with appropriate radii
        waypoints = []
        for x, y, criticality in critical_points[:self.max_waypoints]:
            radius = self._estimate_local_corridor_width(screenshot, x, y)
            waypoints.append({'x': float(x), 'y': float(y), 'radius': radius})
        
        return waypoints
    
    def _create_potential_field(self, screenshot: np.ndarray) -> np.ndarray:
        """Create a potential field from sources to sinks."""
        height, width = screenshot.shape
        potential = np.zeros((height, width), dtype=np.float64)
        
        # Find sources and sinks
        sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
        sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
        
        if not sources or not sinks:
            return potential
        
        # Create distance-based potential field
        for y in range(height):
            for x in range(width):
                if screenshot[y, x] == 1:  # Wall
                    potential[y, x] = -1000.0  # Large negative potential
                else:
                    # Distance to nearest sink (attractive)
                    min_sink_dist = min(math.sqrt((x - sx)**2 + (y - sy)**2) + 1 for sx, sy in sinks)
                    sink_potential = -100.0 / min_sink_dist
                    
                    # Distance to nearest source (repulsive at large distances)
                    min_source_dist = min(math.sqrt((x - sx)**2 + (y - sy)**2) + 1 for sx, sy in sources)
                    source_potential = 50.0 / min_source_dist
                    
                    potential[y, x] = sink_potential + source_potential
        
        # Smooth the field
        for _ in range(3):
            potential = self._smooth_field(potential, screenshot)
        
        return potential
    
    def _smooth_field(self, field: np.ndarray, screenshot: np.ndarray) -> np.ndarray:
        """Smooth the potential field while respecting walls."""
        height, width = field.shape
        smoothed = field.copy()
        
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if screenshot[y, x] != 1:  # Not a wall
                    neighbors = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            ny, nx = y + dy, x + dx
                            if screenshot[ny, nx] != 1:
                                neighbors.append(field[ny, nx])
                    if neighbors:
                        smoothed[y, x] = sum(neighbors) / len(neighbors)
        
        return smoothed
    
    def _find_critical_points(self, potential_field: np.ndarray, screenshot: np.ndarray) -> List[Tuple[int, int, float]]:
        """Find critical points in the potential field."""
        height, width = potential_field.shape
        critical_points = []
        
        for y in range(2, height - 2):
            for x in range(2, width - 2):
                if screenshot[y, x] == 1:  # Skip walls
                    continue
                
                # Calculate gradient magnitude
                grad_x = (potential_field[y, x + 1] - potential_field[y, x - 1]) / 2.0
                grad_y = (potential_field[y + 1, x] - potential_field[y - 1, x]) / 2.0
                grad_magnitude = math.sqrt(grad_x**2 + grad_y**2)
                
                # Look for points with interesting gradient properties
                if grad_magnitude < 0.1:  # Near-zero gradient (critical point)
                    criticality = 10.0 / (grad_magnitude + 0.01)
                    critical_points.append((x, y, criticality))
                elif grad_magnitude > 5.0:  # High gradient (bottleneck)
                    criticality = grad_magnitude * 0.2
                    critical_points.append((x, y, criticality))
        
        # Sort by criticality and return top candidates
        critical_points.sort(key=lambda p: p[2], reverse=True)
        return critical_points[:20]  # Return top 20 candidates
    
    def _estimate_local_corridor_width(self, screenshot: np.ndarray, x: int, y: int) -> float:
        """Estimate corridor width at a point."""
        distances = []
        
        for angle in np.linspace(0, math.pi, 8):  # Check 8 directions
            dx, dy = math.cos(angle), math.sin(angle)
            
            for dist in range(1, 30):
                test_x = int(x + dx * dist)
                test_y = int(y + dy * dist)
                
                if (test_x < 0 or test_x >= screenshot.shape[1] or 
                    test_y < 0 or test_y >= screenshot.shape[0] or
                    screenshot[test_y, test_x] == 1):
                    distances.append(dist)
                    break
        
        if distances:
            return min(distances) * 0.7
        return 5.0


class SwarmIntelligenceGenerator(WaypointGenerator):
    """
    Generate waypoints using swarm intelligence (particle swarm optimization).
    
    Simulates a swarm of particles exploring the space to find optimal waypoint placements.
    """
    
    def __init__(self, num_particles: int = 15, iterations: int = 30, max_waypoints: int = 6):
        super().__init__("SwarmIntelligence")
        self.num_particles = num_particles
        self.iterations = iterations
        self.max_waypoints = max_waypoints
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using particle swarm optimization."""
        height, width = screenshot.shape
        
        # Find passable area bounds
        passable_area = self._get_passable_bounds(screenshot)
        if not passable_area:
            return []
        
        min_x, max_x, min_y, max_y = passable_area
        
        # Initialize swarm
        particles = []
        global_best = None
        global_best_score = float('inf')
        
        for _ in range(self.num_particles):
            particle = self._create_particle(min_x, max_x, min_y, max_y)
            particles.append(particle)
        
        # PSO main loop
        for iteration in range(self.iterations):
            for particle in particles:
                # Evaluate current position
                waypoints = self._particle_to_waypoints(particle)
                score = improved_score_waypoint_list(screenshot, waypoints, penalize_skippable=True)
                
                # Update personal best
                if score < particle['best_score']:
                    particle['best_score'] = score
                    particle['best_position'] = particle['position'].copy()
                
                # Update global best
                if score < global_best_score:
                    global_best_score = score
                    global_best = particle['position'].copy()
            
            # Update particle velocities and positions
            if global_best is not None:
                for particle in particles:
                    self._update_particle(particle, global_best, min_x, max_x, min_y, max_y)
        
        # Return best solution
        if global_best is not None:
            return self._particle_to_waypoints({'position': global_best})
        return []
    
    def _get_passable_bounds(self, screenshot: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Get bounds of passable area."""
        height, width = screenshot.shape
        
        passable_y, passable_x = np.where(screenshot != 1)
        if len(passable_x) == 0:
            return None
        
        return (int(np.min(passable_x)) + 5, int(np.max(passable_x)) - 5,
                int(np.min(passable_y)) + 5, int(np.max(passable_y)) - 5)
    
    def _create_particle(self, min_x: int, max_x: int, min_y: int, max_y: int) -> Dict:
        """Create a particle with random initial position and velocity."""
        # Position encodes waypoint configuration: [num_waypoints, x1, y1, r1, x2, y2, r2, ...]
        position = [random.randint(0, self.max_waypoints)]  # Number of waypoints
        
        for _ in range(self.max_waypoints):
            position.extend([
                random.uniform(min_x, max_x),      # x
                random.uniform(min_y, max_y),      # y
                random.uniform(3.0, 15.0)          # radius
            ])
        
        # Initialize velocity
        velocity = [random.uniform(-2, 2) for _ in range(len(position))]
        
        return {
            'position': position,
            'velocity': velocity,
            'best_position': position.copy(),
            'best_score': float('inf')
        }
    
    def _particle_to_waypoints(self, particle: Dict) -> List[Dict[str, float]]:
        """Convert particle position to waypoints list."""
        position = particle['position']
        num_waypoints = max(0, min(self.max_waypoints, int(position[0])))
        
        waypoints = []
        for i in range(num_waypoints):
            base_idx = 1 + i * 3
            if base_idx + 2 < len(position):
                waypoints.append({
                    'x': float(position[base_idx]),
                    'y': float(position[base_idx + 1]),
                    'radius': max(2.0, float(position[base_idx + 2]))
                })
        
        return waypoints
    
    def _update_particle(self, particle: Dict, global_best: List[float],
                        min_x: int, max_x: int, min_y: int, max_y: int):
        """Update particle velocity and position."""
        w = 0.7  # Inertia weight
        c1 = 1.5  # Cognitive parameter
        c2 = 1.5  # Social parameter
        
        for i in range(len(particle['velocity'])):
            r1, r2 = random.random(), random.random()
            
            # Update velocity
            particle['velocity'][i] = (
                w * particle['velocity'][i] +
                c1 * r1 * (particle['best_position'][i] - particle['position'][i]) +
                c2 * r2 * (global_best[i] - particle['position'][i])
            )
            
            # Update position
            particle['position'][i] += particle['velocity'][i]
            
            # Apply bounds
            if i == 0:  # Number of waypoints
                particle['position'][i] = max(0, min(self.max_waypoints, particle['position'][i]))
            elif i % 3 == 1:  # x coordinate
                particle['position'][i] = max(min_x, min(max_x, particle['position'][i]))
            elif i % 3 == 2:  # y coordinate
                particle['position'][i] = max(min_y, min(max_y, particle['position'][i]))
            else:  # radius
                particle['position'][i] = max(2.0, min(20.0, particle['position'][i]))


class AdaptiveRandomGenerator(WaypointGenerator):
    """
    Generate waypoints using adaptive random sampling with learning.
    
    Uses multiple random attempts but learns from previous attempts to
    guide future sampling towards better regions.
    """
    
    def __init__(self, num_attempts: int = 100, learning_rate: float = 0.1):
        super().__init__("AdaptiveRandom")
        self.num_attempts = num_attempts
        self.learning_rate = learning_rate
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using adaptive random sampling."""
        height, width = screenshot.shape
        
        # Create probability map for waypoint placement
        prob_map = np.ones((height, width), dtype=np.float64)
        prob_map[screenshot == 1] = 0.0  # Can't place on walls
        
        best_waypoints = []
        best_score = float('inf')
        
        for attempt in range(self.num_attempts):
            # Generate random waypoint configuration
            num_waypoints = random.randint(0, 8)
            waypoints = []
            
            for _ in range(num_waypoints):
                # Sample position based on probability map
                y, x = self._sample_from_prob_map(prob_map)
                radius = random.uniform(3.0, 15.0)
                waypoints.append({'x': float(x), 'y': float(y), 'radius': radius})
            
            # Evaluate
            score = improved_score_waypoint_list(screenshot, waypoints, penalize_skippable=True)
            
            # Update best
            if score < best_score:
                best_score = score
                best_waypoints = waypoints.copy()
            
            # Update probability map based on results
            self._update_prob_map(prob_map, waypoints, score)
        
        return best_waypoints
    
    def _sample_from_prob_map(self, prob_map: np.ndarray) -> Tuple[int, int]:
        """Sample a position from the probability map."""
        # Flatten and normalize
        flat_probs = prob_map.flatten()
        if flat_probs.sum() == 0:
            flat_probs = np.ones_like(flat_probs)
        flat_probs = flat_probs / flat_probs.sum()
        
        # Sample
        idx = np.random.choice(len(flat_probs), p=flat_probs)
        y, x = divmod(idx, prob_map.shape[1])
        return y, x
    
    def _update_prob_map(self, prob_map: np.ndarray, waypoints: List[Dict[str, float]], score: float):
        """Update probability map based on waypoint performance."""
        if not waypoints:
            return
        
        # Convert score to feedback (good scores increase probability)
        feedback = math.exp(-score / 100.0)  # Good scores (low) give high feedback
        
        # Update probabilities around waypoint locations
        for wp in waypoints:
            x, y = int(wp['x']), int(wp['y'])
            radius = int(wp['radius']) + 5
            
            for dy in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    ny, nx = y + dy, x + dx
                    if (0 <= ny < prob_map.shape[0] and 0 <= nx < prob_map.shape[1] and
                        prob_map[ny, nx] > 0):
                        
                        distance = math.sqrt(dx**2 + dy**2)
                        if distance <= radius:
                            # Closer points get stronger updates
                            influence = math.exp(-distance / radius)
                            prob_map[ny, nx] = (prob_map[ny, nx] * (1 - self.learning_rate) +
                                              feedback * influence * self.learning_rate)


def create_creative_tournament():
    """Create a tournament with creative algorithms."""
    try:
        from .waypoint_generation import WaypointTournament, NullGenerator, CornerTurningGenerator
    except ImportError:
        from waypoint_generation import WaypointTournament, NullGenerator, CornerTurningGenerator
    
    tournament = WaypointTournament()
    
    # Add basic algorithms
    tournament.add_generator(NullGenerator())
    tournament.add_generator(CornerTurningGenerator())
    
    # Add creative algorithms (with limited parameters for speed)
    tournament.add_generator(GeneticWaypointGenerator(population_size=10, generations=20, max_waypoints=5))
    tournament.add_generator(FlowFieldGenerator(max_waypoints=6))
    tournament.add_generator(SwarmIntelligenceGenerator(num_particles=8, iterations=15, max_waypoints=5))
    tournament.add_generator(AdaptiveRandomGenerator(num_attempts=50))
    
    # Add variations
    tournament.add_generator(GeneticWaypointGenerator(population_size=15, generations=15, max_waypoints=4, mutation_rate=0.5))
    tournament.add_generator(SwarmIntelligenceGenerator(num_particles=12, iterations=20, max_waypoints=4))
    
    return tournament


if __name__ == "__main__":
    # Test creative generators
    print("Testing creative waypoint generators...")
    
    from waypoint_test_runner import create_synthetic_test_cases
    
    test_cases = create_synthetic_test_cases()
    tournament = create_creative_tournament()
    
    # Test on just one case to verify they work
    test_name, screenshot = test_cases[1]  # L-shape
    print(f"\nTesting creative algorithms on {test_name}:")
    
    for generator in tournament.generators[2:4]:  # Test first 2 creative algorithms
        try:
            print(f"Testing {generator.name}...")
            start_time = time.time()
            waypoints = generator.generate_waypoints(screenshot)
            duration = time.time() - start_time
            
            score = improved_score_waypoint_list(screenshot, waypoints, penalize_skippable=True)
            print(f"  Generated {len(waypoints)} waypoints in {duration:.2f}s, score: {score:.2f}")
            
        except Exception as e:
            print(f"  ERROR in {generator.name}: {e}")
    
    print("\nCreative algorithms test completed!")