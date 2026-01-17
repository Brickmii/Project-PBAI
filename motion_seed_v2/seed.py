"""
Motion Calendar Seed - Main Module (v2)

PBAI Functional Stack:
1. Polarity (±) — infinite opposition
2. Imaginary phase (i) — proto-heat  
3. Existence Gate — Euler's Identity: e^(iπ) + 1 = 0
4. Heat (k) — post-existence real magnitude (no subtraction)
5. Righteousness — learnable transform functions
6. Order — structural constraints (extensible)
7. Movement — vector expression (random or chosen)

Agency randomizes; Intelligence chooses.

Architecture:
    Input → Polarity + Phase → Existence Gate → Heat
                                     ↓
    Perception → Kernel (grows) → Righteousness Gate → Identity (self)
                                     ↓
                            Associative Memory → Retrieval
                                     ↓
                            Randomizer (Agency/Intelligence) → Action
"""

import time
import json
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

from core import (
    Particle, DarkParticle, Heat, Polarity, Movement, MovementMode,
    Transform, create_particle_from_input
)
from kernel import Kernel
from identity import Identity, RighteousnessGate, RighteousnessResult
from randomizer import Randomizer, Thought
from retrieval import AssociativeMemory, Retriever


class MotionSeed:
    """
    The complete Motion Calendar Seed (v2).
    
    A grown intelligence based on proper mathematical foundations:
    - Complex heat initialization via Euler's existence gate
    - Polarity as infinite oscillation
    - Learnable transform functions (righteousness)
    - Extensible order constraints
    - Agency vs Intelligence modes
    """
    
    def __init__(self, data_path: str = "./seed_data", mode: MovementMode = MovementMode.AGENCY):
        self.data_path = Path(data_path)
        self.data_path.mkdir(parents=True, exist_ok=True)
        
        self.mode = mode  # Agency or Intelligence
        
        # Core modules
        self.kernel = Kernel(self.data_path / "kernel", mode=mode)
        self.identity = Identity(self.data_path / "identity")
        self.gate = RighteousnessGate()
        self.randomizer = Randomizer(self.identity, mode=mode)
        
        # Memory and retrieval
        self.memory = AssociativeMemory()
        self.retriever = Retriever(self.memory, self.identity)
        self._load_memory()
        
        # Statistics
        self.total_perceptions = 0
        self.total_actions = 0
        self.total_existence_failures = 0
        self.created_at = time.time()
    
    def set_mode(self, mode: MovementMode):
        """Switch between Agency and Intelligence modes."""
        self.mode = mode
        self.kernel.mode = mode
        self.randomizer.set_mode(mode)
    
    # ========== PERCEPTION ==========
    
    def perceive(self, input_data: Any) -> List[Tuple[str, bool, str]]:
        """
        Perception module - process input from environment.
        
        Full pipeline:
        1. Create polarity and imaginary phase
        2. Run existence gate (Euler's Identity)
        3. Apply transform to get righteousness position
        4. Evaluate at righteousness gate
        5. Integrate or reject
        """
        self.total_perceptions += 1
        results = []
        
        # Process statements for strong associations
        if isinstance(input_data, str):
            self.retriever.associate_statement(input_data)
        
        # Tokenize input
        units = self._tokenize(input_data)
        
        for unit in units:
            # Experience in kernel (existence gate happens here)
            particle = self.kernel.experience(unit, mode=self.mode)
            
            # Build associations
            self.memory.observe(particle)
            
            # Evaluate at righteousness gate
            gate_result = self.gate.evaluate(particle, self.identity)
            
            if gate_result.passed:
                # Get counterpart if found
                counterpart = None
                if gate_result.counterpart_id:
                    counterpart = self.kernel.get_particle(gate_result.counterpart_id)
                
                # Integrate into identity
                self.identity.integrate(particle, counterpart)
            
            results.append((
                str(unit),
                gate_result.passed,
                gate_result.reason
            ))
        
        return results
    
    def _tokenize(self, data: Any) -> List[Any]:
        """Break input into processable units."""
        if isinstance(data, str):
            units = []
            
            # Individual characters
            for char in data:
                if char.strip():
                    units.append(char)
            
            # Words
            words = data.split()
            units.extend(words)
            
            # Full input if multi-token
            if len(words) > 1:
                units.append(data)
            
            return units
        
        elif isinstance(data, (list, tuple)):
            units = []
            for item in data:
                units.extend(self._tokenize(item))
            return units
        
        else:
            return [data]
    
    # ========== DARK MATTER ==========
    
    def create_dark(self) -> DarkParticle:
        """Create a dark particle (pre-existence state)."""
        return self.kernel.create_dark_particle()
    
    def attempt_existence(self, dark_id: str) -> Tuple[bool, Optional[Particle]]:
        """Attempt to bring a dark particle into existence."""
        return self.kernel.attempt_dark_existence(dark_id)
    
    # ========== THOUGHT ==========
    
    def think(self, seed: str = None, length: int = 5) -> Thought:
        """Generate a thought through identity (agency mode)."""
        return self.randomizer.think(seed=seed, length=length)
    
    def choose_thought(self, path: List[str]) -> Thought:
        """
        Intelligence mode: choose a specific thought path.
        
        Only works in INTELLIGENCE mode.
        """
        if self.mode != MovementMode.INTELLIGENCE:
            self.set_mode(MovementMode.INTELLIGENCE)
        return self.randomizer.choose_thought(path)
    
    def brainstorm(self, seed: str = None, count: int = 5) -> List[Thought]:
        """Generate multiple thoughts."""
        return self.randomizer.brainstorm(seed=seed, count=count)
    
    # ========== RETRIEVAL ==========
    
    def query(self, question: str) -> str:
        """Answer a question using associative memory."""
        return self.retriever.query(question)
    
    def recall(self, cue: str) -> List[str]:
        """Recall content associated with a cue."""
        return self.memory.recall(cue, self.identity.domains)
    
    def associations(self, term: str) -> List[Tuple[str, float]]:
        """Get raw associations for a term."""
        return self.memory.retrieve(term, top_k=20)
    
    # ========== ACTION ==========
    
    def act(self, thought: Thought = None) -> str:
        """Action module - produce output."""
        self.total_actions += 1
        
        if thought is None:
            thought = self.think()
        
        return thought.to_string()
    
    def respond(self, input_data: Any) -> str:
        """Full perception-thought-action cycle."""
        self.perceive(input_data)
        
        recent_domains = list(self.kernel.domains.keys())[-3:]
        seed = recent_domains[-1] if recent_domains else None
        thought = self.think(seed=seed)
        
        return self.act(thought)
    
    # ========== LEARNING ==========
    
    def learn_transform(self, name: str, func) -> Transform:
        """
        Learn a new transform function.
        
        This is how intelligence acquires new righteousness transforms.
        """
        return self.kernel.learn_transform(name, func)
    
    def add_constraint(self, name: str, check_func):
        """
        Add a new order constraint.
        
        This is how intelligence extends structural constraints.
        """
        return self.kernel.add_constraint(name, check_func)
    
    # ========== STATE ==========
    
    def save(self):
        """Persist all state to disk."""
        self.kernel.save()
        self.identity.save()
        self._save_memory()
    
    def _save_memory(self):
        """Save associative memory."""
        memory_file = self.data_path / "memory.json"
        data = {
            "associations": {},
            "content_index": dict(self.memory.content_index),
            "domain_content": {k: [str(v) for v in vs] for k, vs in self.memory.domain_content.items()},
            "stats": self.memory.get_stats()
        }
        
        for source, targets in self.memory.associations.items():
            data["associations"][source] = {
                target: {
                    "strength": assoc.strength,
                    "co_occurrences": assoc.co_occurrences,
                    "last_activated": assoc.last_activated
                }
                for target, assoc in targets.items()
            }
        
        with open(memory_file, "w") as f:
            json.dump(data, f, indent=2)
    
    def _load_memory(self):
        """Load associative memory."""
        memory_file = self.data_path / "memory.json"
        if not memory_file.exists():
            return
        
        try:
            with open(memory_file, "r") as f:
                data = json.load(f)
            
            for key, pids in data.get("content_index", {}).items():
                self.memory.content_index[key] = pids
            
            for domain_id, contents in data.get("domain_content", {}).items():
                self.memory.domain_content[domain_id] = contents
            
            from retrieval import Association
            for source, targets in data.get("associations", {}).items():
                for target, assoc_data in targets.items():
                    self.memory.associations[source][target] = Association(
                        source=source,
                        target=target,
                        strength=assoc_data["strength"],
                        co_occurrences=assoc_data.get("co_occurrences", 1),
                        last_activated=assoc_data.get("last_activated", time.time())
                    )
                    if source not in self.memory.reverse_index[target]:
                        self.memory.reverse_index[target].append(source)
            
            self.memory.total_associations = sum(
                len(targets) for targets in self.memory.associations.values()
            )
        
        except Exception as e:
            print(f"Error loading memory: {e}")
    
    def get_status(self) -> Dict:
        """Get complete status."""
        return {
            "mode": self.mode.value,
            "uptime": time.time() - self.created_at,
            "total_perceptions": self.total_perceptions,
            "total_actions": self.total_actions,
            "kernel": self.kernel.get_stats(),
            "identity": self.identity.get_stats(),
            "gate": self.gate.get_stats(),
            "randomizer": self.randomizer.get_stats(),
            "memory": self.memory.get_stats()
        }
    
    def describe(self) -> str:
        """Human-readable description."""
        status = self.get_status()
        
        lines = [
            "=== Motion Calendar Seed v2 ===",
            f"Mode: {status['mode'].upper()}",
            f"Uptime: {status['uptime']:.1f}s",
            f"Perceptions: {status['total_perceptions']}",
            f"Actions: {status['total_actions']}",
            "",
            "--- Kernel (What I've Experienced) ---",
            f"  Total particles: {status['kernel']['total_particles']}",
            f"  Dark particles: {status['kernel']['total_dark']}",
            f"  Total domains: {status['kernel']['total_domains']}",
            f"  Total heat: {status['kernel']['total_heat']:.2f}",
            f"  Transforms: {status['kernel']['transforms']}",
            f"  Constraints: {status['kernel']['constraints']}",
            "",
            "--- Identity (Who I Am) ---",
            f"  Integrated particles: {status['identity']['total_integrated']}",
            f"  Active domains: {status['identity']['total_domains']}",
            f"  Total heat: {status['identity']['total_heat']:.2f}",
            f"  Learned rules: {status['identity']['total_rules']}",
            f"  Learned transforms: {status['identity']['learned_transforms']}",
            "",
            "--- Righteousness Gate ---",
            f"  Pass rate: {status['gate']['pass_rate']:.1%}",
            f"  Rejected (redundant): {status['gate']['total_rejected_redundant']}",
            f"  Rejected (contradiction): {status['gate']['total_rejected_contradiction']}",
            f"  Transform failures: {status['gate']['total_transform_failures']}",
            f"  Pending counterparts: {status['gate']['pending_count']}",
            "",
            "--- Memory (Associations) ---",
            f"  Total associations: {status['memory']['total_associations']}",
            f"  Indexed content: {status['memory']['indexed_content']}",
            f"  Retrievals: {status['memory']['total_retrievals']}",
            "",
            "--- Thought ---",
            f"  Thoughts generated: {status['randomizer']['total_thoughts']}",
            f"  Chosen thoughts: {status['randomizer']['total_chosen']}",
            f"  Total activations: {status['randomizer']['total_activations']}",
        ]
        
        return "\n".join(lines)


def create_seed(data_path: str = "./seed_data", 
                mode: MovementMode = MovementMode.AGENCY) -> MotionSeed:
    """Create a new Motion Seed."""
    return MotionSeed(data_path, mode)
