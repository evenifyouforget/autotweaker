"""
Learning and adaptive waypoint generation algorithms.

Implements guided optimization approaches that learn from previous attempts
and improve over time, as suggested in research instructions for 
"non-deterministic, stochastic, or optimizing algorithms" and "guided optimization".
"""

import numpy as np
import random
import math
from typing import List, Dict, Tuple, Optional, Any
import copy
import json
from collections import defaultdict, deque

try:
    from .waypoint_generation import WaypointGenerator
    from .improved_waypoint_scoring import improved_score_waypoint_list
except ImportError:
    from waypoint_generation import WaypointGenerator
    from improved_waypoint_scoring import improved_score_waypoint_list


class ReinforcementLearningGenerator(WaypointGenerator):
    """
    Learns waypoint placement strategies using reinforcement learning principles.
    
    Maintains a Q-table of state-action values and explores/exploits placement strategies.
    """
    
    def __init__(self, learning_rate: float = 0.1, exploration_rate: float = 0.3, 
                 discount_factor: float = 0.9):
        super().__init__("ReinforcementLearning")
        self.learning_rate = learning_rate
        self.exploration_rate = exploration_rate
        self.discount_factor = discount_factor
        self.q_table = defaultdict(lambda: defaultdict(float))
        self.experience_buffer = []
        self.attempts = 0
        
    def _discretize_state(self, screenshot: np.ndarray, existing_waypoints: List[Dict]) -> str:
        """Convert continuous level state to discrete state for Q-learning."""
        height, width = screenshot.shape
        
        # Create state representation
        wall_density = np.sum(screenshot == 1) / (height * width)
        
        sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
        sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
        
        # Discretize key features
        state_features = [
            int(wall_density * 10),  # Wall density (0-10)
            len(sources),            # Number of sources
            len(sinks),             # Number of sinks
            len(existing_waypoints), # Current waypoints
            int(width / 50),        # Level width category
            int(height / 50),       # Level height category
        ]
        
        return "_".join(map(str, state_features))
    
    def _get_possible_actions(self, screenshot: np.ndarray) -> List[Tuple[int, int, float]]:
        """Get possible waypoint placement actions."""
        height, width = screenshot.shape
        actions = []
        
        # Sample potential positions
        for _ in range(20):  # Limited action space
            x = random.randint(5, width - 5)
            y = random.randint(5, height - 5)
            
            # Skip walls
            if screenshot[y, x] == 1:
                continue
            
            # Random radius
            radius = random.uniform(3.0, 15.0)
            actions.append((x, y, radius))
        
        return actions
    
    def _select_action(self, state: str, possible_actions: List[Tuple]) -> Tuple[int, int, float]:
        """Select action using epsilon-greedy strategy."""
        if random.random() < self.exploration_rate or not possible_actions:
            # Explore: random action
            if possible_actions:
                return random.choice(possible_actions)
            else:
                return (50, 50, 8.0)  # Default action
        
        # Exploit: best known action
        best_action = None
        best_q_value = float('-inf')
        
        for action in possible_actions:
            action_key = f"{action[0]}_{action[1]}_{int(action[2])}"
            q_value = self.q_table[state][action_key]
            
            if q_value > best_q_value:
                best_q_value = q_value
                best_action = action
        
        return best_action if best_action else random.choice(possible_actions)
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using reinforcement learning."""
        waypoints = []
        self.attempts += 1
        
        # Gradually decrease exploration rate
        self.exploration_rate = max(0.1, 0.3 - self.attempts * 0.001)
        
        max_waypoints = 3
        for step in range(max_waypoints):
            state = self._discretize_state(screenshot, waypoints)
            possible_actions = self._get_possible_actions(screenshot)
            
            if not possible_actions:
                break
            
            action = self._select_action(state, possible_actions)
            x, y, radius = action
            
            # Create waypoint
            new_waypoint = {'x': float(x), 'y': float(y), 'radius': radius}
            
            # Evaluate if this waypoint improves the score
            test_waypoints = waypoints + [new_waypoint]
            score = improved_score_waypoint_list(screenshot, test_waypoints, penalize_skippable=True)
            
            # Store experience for learning
            action_key = f"{x}_{y}_{int(radius)}"
            reward = -score  # Negative because we want lower scores
            
            self.experience_buffer.append({
                'state': state,
                'action': action_key,
                'reward': reward,
                'next_state': self._discretize_state(screenshot, test_waypoints)
            })
            
            # Update Q-table
            if len(self.experience_buffer) > 1:
                prev_exp = self.experience_buffer[-2]
                current_q = self.q_table[prev_exp['state']][prev_exp['action']]
                
                # Q-learning update
                max_next_q = max(self.q_table[prev_exp['next_state']].values()) if self.q_table[prev_exp['next_state']] else 0
                new_q = current_q + self.learning_rate * (prev_exp['reward'] + self.discount_factor * max_next_q - current_q)
                
                self.q_table[prev_exp['state']][prev_exp['action']] = new_q
            
            # Accept waypoint if it improves score or randomly for exploration
            if score < 1000 or random.random() < 0.3:  # Accept good waypoints or explore
                waypoints.append(new_waypoint)
            else:
                break  # Stop if waypoint makes things much worse
        
        # Limit experience buffer size
        if len(self.experience_buffer) > 1000:
            self.experience_buffer = self.experience_buffer[-500:]
        
        return waypoints


class AdaptiveTemplateGenerator(WaypointGenerator):
    """
    Learns successful waypoint patterns and adapts them to new levels.
    
    Builds a library of successful waypoint templates and applies similar
    patterns to levels with similar characteristics.
    """
    
    def __init__(self):
        super().__init__("AdaptiveTemplate")
        self.successful_templates = []
        self.level_features = []
        
    def _extract_level_features(self, screenshot: np.ndarray) -> Dict[str, float]:
        """Extract features that characterize a level."""
        height, width = screenshot.shape
        
        # Basic features
        wall_ratio = np.sum(screenshot == 1) / (height * width)
        sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
        sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
        
        # Geometric features
        if sources and sinks:
            # Average distance from sources to sinks
            distances = []
            for sx, sy in sources:
                min_dist = min(math.sqrt((sx-tx)**2 + (sy-ty)**2) for tx, ty in sinks)
                distances.append(min_dist)
            avg_distance = np.mean(distances)
            
            # Level span
            all_points = sources + sinks
            x_span = max(p[0] for p in all_points) - min(p[0] for p in all_points)
            y_span = max(p[1] for p in all_points) - min(p[1] for p in all_points)
        else:
            avg_distance = 0
            x_span = y_span = 0
        
        return {
            'wall_ratio': wall_ratio,
            'source_count': len(sources),
            'sink_count': len(sinks),
            'avg_source_sink_distance': avg_distance / max(width, height),  # Normalized
            'aspect_ratio': width / height,
            'x_span_ratio': x_span / width,
            'y_span_ratio': y_span / height,
            'level_size': width * height
        }
    
    def _find_similar_template(self, level_features: Dict[str, float]) -> Optional[Dict]:
        """Find the most similar successful template."""
        if not self.successful_templates:
            return None
        
        best_template = None
        best_similarity = -1
        
        for template in self.successful_templates:
            # Calculate similarity using weighted feature differences
            similarity = 0
            weights = {
                'wall_ratio': 2.0,
                'source_count': 1.0,
                'sink_count': 1.0,
                'avg_source_sink_distance': 1.5,
                'aspect_ratio': 1.0
            }
            
            total_weight = 0
            for feature, weight in weights.items():
                if feature in level_features and feature in template['features']:
                    diff = abs(level_features[feature] - template['features'][feature])
                    similarity += weight * (1.0 - min(1.0, diff))
                    total_weight += weight
            
            if total_weight > 0:
                similarity /= total_weight
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_template = template
        
        # Only use template if similarity is reasonable
        return best_template if best_similarity > 0.6 else None
    
    def _adapt_template_to_level(self, template: Dict, screenshot: np.ndarray, 
                                level_features: Dict[str, float]) -> List[Dict[str, float]]:
        """Adapt a successful template to the current level."""
        height, width = screenshot.shape
        adapted_waypoints = []
        
        # Scale template waypoints to current level
        template_features = template['features']
        waypoints = template['waypoints']
        
        # Calculate scaling factors
        size_scale = math.sqrt(level_features['level_size'] / template_features['level_size'])
        x_scale = level_features['x_span_ratio'] / max(0.1, template_features['x_span_ratio'])
        y_scale = level_features['y_span_ratio'] / max(0.1, template_features['y_span_ratio'])
        
        # Find current sources/sinks for reference
        sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
        sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
        
        if not sources or not sinks:
            return []
        
        # Reference points for adaptation
        ref_source = sources[0] if sources else (width//4, height//2)
        ref_sink = sinks[0] if sinks else (3*width//4, height//2)
        
        for wp in waypoints:
            # Scale position relative to level characteristics
            scaled_x = ref_source[0] + (wp['x'] - template_features.get('ref_x', 50)) * x_scale
            scaled_y = ref_source[1] + (wp['y'] - template_features.get('ref_y', 50)) * y_scale
            scaled_radius = wp['radius'] * size_scale
            
            # Ensure waypoint is within bounds and not in a wall
            scaled_x = max(5, min(width-5, scaled_x))
            scaled_y = max(5, min(height-5, scaled_y))
            scaled_radius = max(2, min(20, scaled_radius))
            
            # Check if position is valid (not in wall)
            if screenshot[int(scaled_y), int(scaled_x)] != 1:
                adapted_waypoints.append({
                    'x': float(scaled_x),
                    'y': float(scaled_y),
                    'radius': float(scaled_radius)
                })
        
        return adapted_waypoints
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using adaptive template matching."""
        level_features = self._extract_level_features(screenshot)
        
        # Try to find and adapt similar template
        similar_template = self._find_similar_template(level_features)
        
        if similar_template:
            waypoints = self._adapt_template_to_level(similar_template, screenshot, level_features)
            if waypoints:
                return waypoints
        
        # Fallback: generate new waypoints using simple strategy
        sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
        sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
        
        if not sources or not sinks:
            return []
        
        # Place waypoint midway between source and sink
        source = sources[0]
        sink = sinks[0] 
        
        mid_x = (source[0] + sink[0]) / 2
        mid_y = (source[1] + sink[1]) / 2
        
        # Adjust if position is in wall
        height, width = screenshot.shape
        for offset in range(10):
            for dx, dy in [(0,0), (offset,0), (-offset,0), (0,offset), (0,-offset)]:
                test_x = int(mid_x + dx)
                test_y = int(mid_y + dy)
                
                if (0 <= test_x < width and 0 <= test_y < height and 
                    screenshot[test_y, test_x] != 1):
                    return [{'x': float(test_x), 'y': float(test_y), 'radius': 8.0}]
        
        return []
    
    def learn_from_success(self, screenshot: np.ndarray, successful_waypoints: List[Dict[str, float]], score: float):
        """Learn from a successful waypoint configuration."""
        if score < 500:  # Only learn from good solutions
            level_features = self._extract_level_features(screenshot)
            sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
            
            if sources:
                level_features['ref_x'] = sources[0][0]
                level_features['ref_y'] = sources[0][1]
            
            template = {
                'features': level_features,
                'waypoints': copy.deepcopy(successful_waypoints),
                'score': score
            }
            
            self.successful_templates.append(template)
            
            # Limit template library size
            if len(self.successful_templates) > 50:
                # Keep best templates
                self.successful_templates.sort(key=lambda t: t['score'])
                self.successful_templates = self.successful_templates[:30]


class EvolutionaryStrategyGenerator(WaypointGenerator):
    """
    Uses evolutionary strategies to evolve waypoint placement parameters.
    
    Maintains a population of waypoint generation strategies and evolves
    them based on performance feedback.
    """
    
    def __init__(self, population_size: int = 20):
        super().__init__("EvolutionaryStrategy")
        self.population_size = population_size
        self.population = []
        self.generation = 0
        self._initialize_population()
    
    def _initialize_population(self):
        """Initialize population with random strategies."""
        for _ in range(self.population_size):
            strategy = {
                'bias_toward_center': random.uniform(0.0, 1.0),
                'prefer_corners': random.uniform(0.0, 1.0),
                'avoid_walls_distance': random.uniform(5.0, 20.0),
                'radius_factor': random.uniform(0.5, 2.0),
                'max_waypoints': random.randint(1, 5),
                'clustering_tendency': random.uniform(0.0, 1.0),
                'fitness': 0.0,
                'evaluations': 0
            }
            self.population.append(strategy)
    
    def _select_strategy(self) -> Dict:
        """Select strategy using tournament selection."""
        tournament_size = min(3, len(self.population))
        candidates = random.sample(self.population, tournament_size)
        
        # Select based on average fitness (lower is better for scores)
        best = min(candidates, key=lambda s: s['fitness'] / max(1, s['evaluations']))
        return best
    
    def _mutate_strategy(self, strategy: Dict) -> Dict:
        """Create mutated version of strategy."""
        new_strategy = copy.deepcopy(strategy)
        
        # Mutation parameters
        mutation_rate = 0.1
        mutation_strength = 0.2
        
        if random.random() < mutation_rate:
            new_strategy['bias_toward_center'] = max(0, min(1, 
                new_strategy['bias_toward_center'] + random.gauss(0, mutation_strength)))
        
        if random.random() < mutation_rate:
            new_strategy['prefer_corners'] = max(0, min(1,
                new_strategy['prefer_corners'] + random.gauss(0, mutation_strength)))
        
        if random.random() < mutation_rate:
            new_strategy['avoid_walls_distance'] = max(2, min(30,
                new_strategy['avoid_walls_distance'] + random.gauss(0, 3)))
        
        if random.random() < mutation_rate:
            new_strategy['radius_factor'] = max(0.2, min(3,
                new_strategy['radius_factor'] + random.gauss(0, 0.3)))
        
        if random.random() < mutation_rate:
            new_strategy['max_waypoints'] = max(1, min(6,
                new_strategy['max_waypoints'] + random.randint(-1, 1)))
        
        # Reset fitness for new strategy
        new_strategy['fitness'] = 0.0
        new_strategy['evaluations'] = 0
        
        return new_strategy
    
    def _apply_strategy(self, strategy: Dict, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using given strategy."""
        height, width = screenshot.shape
        waypoints = []
        
        sources = [(x, y) for y, x in zip(*np.where(screenshot == 3))]
        sinks = [(x, y) for y, x in zip(*np.where(screenshot == 4))]
        
        if not sources or not sinks:
            return []
        
        center_x, center_y = width // 2, height // 2
        
        for _ in range(strategy['max_waypoints']):
            attempts = 0
            while attempts < 20:
                attempts += 1
                
                # Generate candidate position based on strategy
                if random.random() < strategy['bias_toward_center']:
                    # Bias toward center
                    x = random.gauss(center_x, width * 0.3)
                    y = random.gauss(center_y, height * 0.3)
                else:
                    # Uniform random
                    x = random.uniform(0, width)
                    y = random.uniform(0, height)
                
                # Apply corner preference
                if random.random() < strategy['prefer_corners']:
                    # Bias toward corners/walls
                    if random.random() < 0.5:
                        x = random.choice([random.uniform(0, width*0.2), random.uniform(width*0.8, width)])
                    if random.random() < 0.5:
                        y = random.choice([random.uniform(0, height*0.2), random.uniform(height*0.8, height)])
                
                x = max(5, min(width-5, x))
                y = max(5, min(height-5, y))
                
                # Check wall avoidance
                if screenshot[int(y), int(x)] == 1:
                    continue
                
                # Check distance to walls
                min_wall_dist = strategy['avoid_walls_distance']
                too_close_to_wall = False
                
                for dx in range(-int(min_wall_dist), int(min_wall_dist)+1):
                    for dy in range(-int(min_wall_dist), int(min_wall_dist)+1):
                        nx, ny = int(x + dx), int(y + dy)
                        if (0 <= nx < width and 0 <= ny < height and 
                            screenshot[ny, nx] == 1 and 
                            math.sqrt(dx*dx + dy*dy) < min_wall_dist):
                            too_close_to_wall = True
                            break
                    if too_close_to_wall:
                        break
                
                if too_close_to_wall:
                    continue
                
                # Calculate radius
                base_radius = min(width, height) * 0.05 * strategy['radius_factor']
                radius = max(3, min(20, base_radius))
                
                waypoint = {'x': float(x), 'y': float(y), 'radius': radius}
                waypoints.append(waypoint)
                break
        
        return waypoints
    
    def generate_waypoints(self, screenshot: np.ndarray) -> List[Dict[str, float]]:
        """Generate waypoints using evolutionary strategy."""
        # Select and apply strategy
        strategy = self._select_strategy()
        waypoints = self._apply_strategy(strategy, screenshot)
        
        # Evaluate and update strategy fitness
        score = improved_score_waypoint_list(screenshot, waypoints, penalize_skippable=True)
        
        strategy['fitness'] += score
        strategy['evaluations'] += 1
        
        # Evolve population periodically
        self.generation += 1
        if self.generation % 50 == 0:
            self._evolve_population()
        
        return waypoints
    
    def _evolve_population(self):
        """Evolve the population by replacing worst performers."""
        # Sort by average fitness (lower is better)
        self.population.sort(key=lambda s: s['fitness'] / max(1, s['evaluations']))
        
        # Replace worst 25% with mutated versions of better strategies
        replace_count = self.population_size // 4
        for i in range(replace_count):
            # Select good strategy to mutate
            parent_idx = random.randint(0, self.population_size // 2)
            parent = self.population[parent_idx]
            
            # Replace worst performer
            worst_idx = -(i + 1)
            self.population[worst_idx] = self._mutate_strategy(parent)


def create_learning_tournament():
    """Create tournament with learning algorithms."""
    try:
        from .waypoint_generation import WaypointTournament
    except ImportError:
        from waypoint_generation import WaypointTournament
    
    tournament = WaypointTournament()
    
    # Add learning algorithms
    tournament.add_generator(ReinforcementLearningGenerator())
    tournament.add_generator(AdaptiveTemplateGenerator())
    tournament.add_generator(EvolutionaryStrategyGenerator(population_size=15))
    
    return tournament


if __name__ == "__main__":
    # Test learning algorithms
    print("Testing learning waypoint algorithms...")
    
    from waypoint_test_runner import create_synthetic_test_cases
    
    test_cases = create_synthetic_test_cases()
    tournament = create_learning_tournament()
    
    print(f"Created learning tournament with {len(tournament.generators)} algorithms")
    
    # Test on L-shape case multiple times to see learning
    test_name, screenshot = test_cases[1]
    print(f"\nTesting learning algorithms on {test_name} (multiple rounds):")
    
    for round_num in range(1, 4):
        print(f"\n--- Round {round_num} ---")
        
        for generator in tournament.generators:
            try:
                import time
                start_time = time.time()
                waypoints = generator.generate_waypoints(screenshot)
                duration = time.time() - start_time
                
                score = improved_score_waypoint_list(screenshot, waypoints, 
                                                   penalize_skippable=True)
                
                print(f"  {generator.name:20} | {len(waypoints):2d} waypoints | {duration:5.3f}s | Score: {score:8.2f}")
                
                # Let adaptive template learn from good results
                if hasattr(generator, 'learn_from_success') and score < 400:
                    generator.learn_from_success(screenshot, waypoints, score)
                
            except Exception as e:
                print(f"  {generator.name:20} | ERROR: {e}")
    
    print("\nLearning algorithms test completed!")