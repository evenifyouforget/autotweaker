from dataclasses import dataclass
from collections import namedtuple
import random
from subprocess import Popen
from typing import Any, List, Optional
from get_design import FCDesignStruct
from .measure_design import measure_design
from .mutate_validate import normalize_design, generate_mutant

GardenStatus = namedtuple('GardenStatus', ['num_active_threads', 'best_score'])

@dataclass
class Creature:
    """
    Represents one candidate variation of a design, which we are testing.
    """
    design_struct: FCDesignStruct
    proc: Optional[Popen] = None
    best_score: Optional[float] = None
    done: bool = False
    
    def __lt__(self, other):
        def key_func(score):
            if score is None:
                return (1, 0)
            return (0, score)
        return key_func(self.best_score) < key_func(other.best_score)
    
    def checkup(self) -> bool:
        """
        Check if the subprocess is done.
        If it is, set the flag and try to fill in the score.
        Return true if currently running.
        """
        if self.proc is None:
            return False
        if self.proc.poll() is not None:
            self.done = True
            try:
                _, _, score = self.proc.stdout.read().strip().split()
                self.best_score = float(score)
            except ValueError:
                pass
            return False
        return True
    
    def start(self, job_config) -> bool:
        """
        Allow it to start, if it is not already started.
        Return True if it just started.
        """
        if self.proc is None:
            self.proc = measure_design(design_struct=self.design_struct, config_data=job_config, nonblocking=True)
            return True
        return False

@dataclass
class Garden:
    """
    Represents a pool of creatures that we are trying to improve.
    """
    creatures: List[Creature]
    max_garden_size: int
    job_config: Any # json
    num_kills: int = 0
    
    def checkup(self) -> GardenStatus:
        """
        Check on the garden, but don't change anything.
        """
        num_active_threads = 0
        best_score = 1e300
        for creature in self.creatures:
            num_active_threads += creature.checkup()
            if creature.best_score is not None:
                best_score = min(best_score, creature.best_score)
        return GardenStatus(
            num_active_threads=num_active_threads,
            best_score=best_score
        )
    
    def start(self, max_new_threads: int):
        """
        Start new threads up to the limit.
        """
        for creature in self.creatures:
            if max_new_threads <= 0:
                return
            max_new_threads -= creature.start(self.job_config)
    
    def evolve(self, max_new_creatures: int, retain_best_k: int):
        """
        Perform a round of garden maintenance.
        - Sort all creatures
        - Try to cull excess creatures, starting with the worst scoring finished ones
        - Try to produce new creatures up to the limit
        """
        # Sort phase
        self.creatures.sort()
        # Cull phase
        target_creatures = self.max_garden_size - max_new_creatures
        num_to_kill = len(self.creatures) - target_creatures
        for index in range(retain_best_k, len(self.creatures))[::-1]:
            if num_to_kill <= 0:
                break
            if self.creatures[index].done:
                self.creatures.pop(index)
                num_to_kill -= 1
                self.num_kills += 1
        # Reproduce phase
        max_new_creatures = min(max_new_creatures, self.max_garden_size - len(self.creatures))
        for _ in range(max_new_creatures):
            parent = random.choice(self.creatures)
            child = generate_mutant(parent.design_struct)
            normalize_result = normalize_design(child)
            if not normalize_design.is_valid:
                continue
            self.creatures.append(Creature(normalize_result.design))