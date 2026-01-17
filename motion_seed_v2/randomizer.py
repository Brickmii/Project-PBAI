"""
Motion Calendar Seed - Randomizer (Thought) Module (v2)
Generates expression through identity constraints.

Two modes:
- Agency: random movement (fires randomly through constraints)
- Intelligence: random OR chosen movement (can also choose)
"""

import random
import math
from typing import List, Optional, Tuple, Any
from dataclasses import dataclass

from core import Particle, Movement, MovementMode
from identity import Identity


@dataclass
class Thought:
    """A generated thought - activation pattern across domains."""
    activations: List[Particle]
    domains_visited: List[str]
    total_heat: float
    coherence: float
    mode: MovementMode
    chosen: bool = False  # Was this thought chosen (vs random)?
    
    def to_string(self) -> str:
        if not self.activations:
            return ""
        return " ".join(str(p.raw_value) for p in self.activations if p.raw_value)
    
    def to_values(self) -> List[Any]:
        return [p.raw_value for p in self.activations if p.raw_value]


class Randomizer:
    """
    Thought generation through identity.
    
    Agency: fires random activations conditioned by identity
    Intelligence: can ALSO choose specific activation paths
    
    Never writes to identity directly.
    Must be promoted through righteousness gate.
    """
    
    def __init__(self, identity: Identity, mode: MovementMode = MovementMode.AGENCY):
        self.identity = identity
        self.mode = mode
        
        # Generation parameters
        self.max_chain_length = 10
        self.correlation_threshold = 0.2
        self.temperature = 1.0  # Higher = more random
        
        # Statistics
        self.total_thoughts = 0
        self.total_activations = 0
        self.total_chosen = 0  # Intelligence-chosen thoughts
    
    def think(self, seed: str = None, length: int = 5, 
              choose_path: List[str] = None) -> Thought:
        """
        Generate a thought.
        
        Args:
            seed: Optional domain to start from
            length: Desired thought length
            choose_path: Intelligence can specify exact domain path
        
        Returns:
            Generated Thought
        """
        if not self.identity.domains:
            return Thought([], [], 0.0, 0.0, self.mode)
        
        self.total_thoughts += 1
        chosen = False
        
        activations = []
        domains_visited = []
        total_heat = 0.0
        
        # Intelligence mode: can choose specific path
        if choose_path and self.mode == MovementMode.INTELLIGENCE:
            chosen = True
            self.total_chosen += 1
            return self._follow_chosen_path(choose_path)
        
        # Choose starting domain
        if seed and seed in self.identity.domains:
            current_domain = seed
        else:
            current_domain = random.choice(list(self.identity.domains.keys()))
        
        # Generate chain
        for _ in range(min(length, self.max_chain_length)):
            domain = self.identity.domains.get(current_domain)
            if not domain or not domain.particles:
                break
            
            particle = self._select_particle(domain)
            if particle:
                activations.append(particle)
                domains_visited.append(current_domain)
                total_heat += particle.heat.k
                self.total_activations += 1
            
            next_domain = self._select_next_domain(current_domain)
            if not next_domain:
                break
            current_domain = next_domain
        
        coherence = self._compute_coherence(activations, domains_visited)
        
        return Thought(
            activations=activations,
            domains_visited=domains_visited,
            total_heat=total_heat,
            coherence=coherence,
            mode=self.mode,
            chosen=chosen
        )
    
    def _follow_chosen_path(self, path: List[str]) -> Thought:
        """Intelligence follows a chosen domain path."""
        activations = []
        domains_visited = []
        total_heat = 0.0
        
        for domain_id in path:
            domain = self.identity.domains.get(domain_id)
            if domain and domain.particles:
                # Select best particle (highest heat)
                best = max(domain.particles.values(), key=lambda p: p.heat.k)
                activations.append(best)
                domains_visited.append(domain_id)
                total_heat += best.heat.k
                self.total_activations += 1
        
        coherence = self._compute_coherence(activations, domains_visited)
        
        return Thought(
            activations=activations,
            domains_visited=domains_visited,
            total_heat=total_heat,
            coherence=coherence,
            mode=MovementMode.INTELLIGENCE,
            chosen=True
        )
    
    def _select_particle(self, domain) -> Optional[Particle]:
        """Select particle from domain."""
        if not domain.particles:
            return None
        
        particles = list(domain.particles.values())
        
        if self.temperature >= 10.0:
            return random.choice(particles)
        
        # Weight by heat
        heats = [p.heat.k for p in particles]
        total_heat = sum(heats)
        
        if total_heat == 0:
            return random.choice(particles)
        
        weights = [h / total_heat for h in heats]
        if self.temperature != 1.0:
            weights = [w ** (1.0 / self.temperature) for w in weights]
            total = sum(weights)
            weights = [w / total for w in weights]
        
        r = random.random()
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return particles[i]
        
        return particles[-1]
    
    def _select_next_domain(self, current_domain: str) -> Optional[str]:
        """Select next domain based on correlations and rules."""
        candidates = []
        weights = []
        
        # Get correlated domains
        correlated = self.identity.get_correlated_domains(
            current_domain, 
            self.correlation_threshold
        )
        for domain_id, strength in correlated:
            if domain_id in self.identity.domains:
                candidates.append(domain_id)
                weights.append(strength)
        
        # Get successor domains
        successors = self.identity.get_successor_domains(current_domain)
        for domain_id, strength in successors:
            if domain_id in self.identity.domains:
                if domain_id not in candidates:
                    candidates.append(domain_id)
                    weights.append(strength * 1.5)
                else:
                    idx = candidates.index(domain_id)
                    weights[idx] += strength * 1.5
        
        if not candidates:
            available = [d for d in self.identity.domains.keys() if d != current_domain]
            return random.choice(available) if available else None
        
        if self.temperature >= 10.0:
            return random.choice(candidates)
        
        total = sum(weights)
        if total == 0:
            return random.choice(candidates)
        
        weights = [w / total for w in weights]
        if self.temperature != 1.0:
            weights = [w ** (1.0 / self.temperature) for w in weights]
            total = sum(weights)
            weights = [w / total for w in weights]
        
        r = random.random()
        cumulative = 0.0
        for i, w in enumerate(weights):
            cumulative += w
            if r <= cumulative:
                return candidates[i]
        
        return candidates[-1]
    
    def _compute_coherence(self, activations: List[Particle], domains: List[str]) -> float:
        """Compute thought coherence."""
        if len(activations) < 2:
            return 1.0
        
        supported = 0
        for i in range(len(domains) - 1):
            from_domain = domains[i]
            to_domain = domains[i + 1]
            
            correlated = dict(self.identity.get_correlated_domains(from_domain))
            if to_domain in correlated:
                supported += correlated[to_domain]
            
            successors = dict(self.identity.get_successor_domains(from_domain))
            if to_domain in successors:
                supported += successors[to_domain]
        
        max_possible = len(domains) - 1
        return supported / max_possible if max_possible > 0 else 1.0
    
    def brainstorm(self, seed: str = None, count: int = 5) -> List[Thought]:
        """Generate multiple thoughts."""
        thoughts = []
        for _ in range(count):
            thought = self.think(seed=seed)
            thoughts.append(thought)
        return sorted(thoughts, key=lambda t: -t.coherence)
    
    def choose_thought(self, path: List[str]) -> Thought:
        """
        Intelligence explicitly chooses a thought path.
        
        Only available in INTELLIGENCE mode.
        """
        if self.mode != MovementMode.INTELLIGENCE:
            raise ValueError("Can only choose thoughts in INTELLIGENCE mode")
        
        return self.think(choose_path=path)
    
    def set_mode(self, mode: MovementMode):
        """Switch between agency and intelligence modes."""
        self.mode = mode
    
    def get_stats(self) -> dict:
        return {
            "total_thoughts": self.total_thoughts,
            "total_activations": self.total_activations,
            "total_chosen": self.total_chosen,
            "temperature": self.temperature,
            "mode": self.mode.value
        }
