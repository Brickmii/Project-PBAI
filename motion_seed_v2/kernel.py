"""
Motion Calendar Seed - Kernel Module (v2)
The Seed that always grows with environment and novelty.

Uses proper mathematical foundations:
- Complex heat initialization
- Existence gate (Euler's Identity)
- Polarity as oscillation
- Agency vs Intelligence modes
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from collections import defaultdict

from core import (
    Particle, DarkParticle, Heat, Polarity, Movement, MovementMode,
    Transform, OrderConstraint, DEFAULT_TRANSFORMS, DEFAULT_CONSTRAINTS,
    create_particle_from_input
)


class TransformLibrary:
    """
    Library of learned transforms.
    
    Righteousness = learnable transform functions.
    Intelligence can acquire new transforms.
    """
    
    def __init__(self):
        self.transforms: Dict[str, Transform] = {}
        # Initialize with defaults
        for t in DEFAULT_TRANSFORMS:
            self.transforms[t.name] = t
    
    def get(self, name: str) -> Optional[Transform]:
        return self.transforms.get(name)
    
    def add(self, transform: Transform):
        """Add a new learned transform."""
        self.transforms[transform.name] = transform
    
    def best_for(self, k: float) -> Transform:
        """Get the most stable transform for given heat."""
        # Sort by stability, return best
        sorted_transforms = sorted(
            self.transforms.values(),
            key=lambda t: t.stability,
            reverse=True
        )
        return sorted_transforms[0] if sorted_transforms else DEFAULT_TRANSFORMS[0]
    
    def all(self) -> List[Transform]:
        return list(self.transforms.values())


class OrderSystem:
    """
    System of structural constraints.
    
    Order constrains and extends valid transforms.
    Intelligence can create new constraints.
    """
    
    def __init__(self):
        self.constraints: Dict[str, OrderConstraint] = {}
        # Initialize with defaults
        for c in DEFAULT_CONSTRAINTS:
            self.constraints[c.name] = c
    
    def add(self, constraint: OrderConstraint):
        """Add a new structural constraint."""
        self.constraints[constraint.name] = constraint
    
    def validate_transition(self, before: Any, after: Any) -> Tuple[bool, List[str]]:
        """
        Check if transition is valid under all constraints.
        
        Returns: (valid, list_of_failed_constraints)
        """
        failed = []
        for name, constraint in self.constraints.items():
            if not constraint.validate(before, after):
                failed.append(name)
        
        return len(failed) == 0, failed
    
    def all(self) -> List[OrderConstraint]:
        return list(self.constraints.values())


class Domain:
    """
    A righteousness map - a plane for storing correlated particles.
    """
    
    def __init__(self, id: str, name: str):
        self.id = id
        self.name = name
        self.particles: Dict[str, Particle] = {}
        self.created_at = time.time()
        
        # Running statistics
        self.mean_x: float = 0.0
        self.mean_y: float = 0.0
        self.var_x: float = 1.0
        self.var_y: float = 1.0
        self.count: int = 0
        self.total_heat: float = 0.0
        
        # Correlations to other domains
        self.correlations: Dict[str, float] = {}
    
    def add_particle(self, particle: Particle):
        """Add particle and update running stats."""
        particle.domain_id = self.id
        self.particles[particle.id] = particle
        
        # Update running statistics (Welford's algorithm)
        self.count += 1
        delta_x = particle.x - self.mean_x
        delta_y = particle.y - self.mean_y
        self.mean_x += delta_x / self.count
        self.mean_y += delta_y / self.count
        
        if self.count > 1:
            self.var_x += delta_x * (particle.x - self.mean_x)
            self.var_y += delta_y * (particle.y - self.mean_y)
        
        self.total_heat += particle.heat.k
    
    def contains_content(self, content: str) -> bool:
        """Check if domain already contains this content."""
        content_lower = content.lower()
        for p in self.particles.values():
            if str(p.raw_value).lower() == content_lower:
                return True
        return False
    
    def get_variance(self) -> Tuple[float, float]:
        if self.count < 2:
            return (1.0, 1.0)
        return (self.var_x / (self.count - 1), self.var_y / (self.count - 1))


class Kernel:
    """
    The Seed - always grows with environment and novelty.
    
    Properly implements:
    - Complex heat initialization via existence gate
    - Polarity as infinite oscillation
    - Agency vs Intelligence modes
    - Transform library (learnable righteousness)
    - Order system (extensible constraints)
    """
    
    def __init__(self, storage_path: Optional[Path] = None, mode: MovementMode = MovementMode.AGENCY):
        self.storage_path = storage_path
        self.mode = mode  # Default operation mode
        
        # Core components
        self.transforms = TransformLibrary()
        self.order = OrderSystem()
        
        # All domains
        self.domains: Dict[str, Domain] = {}
        
        # Dark matter pool (particles that haven't achieved existence)
        self.dark_pool: Dict[str, DarkParticle] = {}
        
        # Global statistics
        self.total_particles = 0
        self.total_dark = 0
        self.total_experiences = 0
        self.total_heat = Heat(k=0.0)
        self.created_at = time.time()
        
        # Order tracking
        self.order_history: List[str] = []
        self.order_window = 100
        
        # Load existing state
        if storage_path:
            self.load()
    
    def experience(self, value: Any, domain_hint: str = None, 
                   mode: MovementMode = None) -> Particle:
        """
        Experience a new input. Always grows the kernel.
        
        The full pipeline:
        1. Create imaginary phase with polarity
        2. Run existence gate (Euler's Identity)
        3. Apply transform to get righteousness position
        4. Generate movement (random for agency, choosable for intelligence)
        5. Store in appropriate domain
        """
        self.total_experiences += 1
        use_mode = mode or self.mode
        
        # Create particle through existence gate
        particle = create_particle_from_input(value, use_mode)
        
        # Determine domain
        domain_id = self._resolve_domain(value, domain_hint)
        
        # Get or create domain
        if domain_id not in self.domains:
            self.domains[domain_id] = Domain(id=domain_id, name=domain_hint or domain_id)
        
        domain = self.domains[domain_id]
        
        # Set domain and raw value
        particle.domain_id = domain_id
        particle.raw_value = value
        
        # Link to order history
        if self.order_history:
            # Validate order constraint
            last_id = self.order_history[-1]
            last_particle = self.get_particle(last_id)
            if last_particle:
                valid, _ = self.order.validate_transition(last_particle, particle)
                if valid:
                    particle.order_links = self.order_history[-3:]
        
        # Add to domain
        domain.add_particle(particle)
        self.total_particles += 1
        self.total_heat = self.total_heat.accumulate(particle.heat.k)
        
        # Update order history
        self.order_history.append(particle.id)
        if len(self.order_history) > self.order_window:
            self.order_history.pop(0)
        
        # Record transform success
        transform = self.transforms.get(particle.transform_used)
        if transform:
            transform.record_success()
        
        return particle
    
    def create_dark_particle(self) -> DarkParticle:
        """
        Create a dark particle (pre-existence state).
        
        Has polarity and imaginary phase, but no real heat yet.
        """
        dark = DarkParticle.create()
        self.dark_pool[dark.id] = dark
        self.total_dark += 1
        return dark
    
    def attempt_dark_existence(self, dark_id: str) -> Tuple[bool, Optional[Particle]]:
        """
        Attempt to bring a dark particle into existence.
        """
        if dark_id not in self.dark_pool:
            return False, None
        
        dark = self.dark_pool[dark_id]
        succeeded, particle, _ = dark.attempt_existence()
        
        if succeeded and particle:
            # Remove from dark pool
            del self.dark_pool[dark_id]
            self.total_dark -= 1
            
            # Add to a domain
            domain_id = "emerged"
            if domain_id not in self.domains:
                self.domains[domain_id] = Domain(id=domain_id, name="emerged_from_dark")
            
            self.domains[domain_id].add_particle(particle)
            self.total_particles += 1
            self.total_heat = self.total_heat.accumulate(particle.heat.k)
            
            return True, particle
        
        return False, None
    
    def _resolve_domain(self, value: Any, hint: str = None) -> str:
        """Determine which domain a value belongs to."""
        if hint:
            return hint
        
        if isinstance(value, str):
            if len(value) == 1:
                if value.isalpha():
                    return f"letter_{value.lower()}"
                elif value.isdigit():
                    return f"digit_{value}"
                else:
                    return f"char_{ord(value)}"
            elif value.isalpha():
                return "word"
            elif value.replace(" ", "").isalpha():
                return "phrase"
            else:
                return "text"
        elif isinstance(value, (int, float)):
            return "number"
        elif isinstance(value, bool):
            return "boolean"
        else:
            return "object"
    
    def get_domain(self, domain_id: str) -> Optional[Domain]:
        return self.domains.get(domain_id)
    
    def get_particle(self, particle_id: str) -> Optional[Particle]:
        """Find a particle by ID across all domains."""
        for domain in self.domains.values():
            if particle_id in domain.particles:
                return domain.particles[particle_id]
        return None
    
    def learn_transform(self, name: str, func) -> Transform:
        """
        Learn a new transform function.
        
        This is how intelligence acquires new righteousness transforms.
        """
        transform = Transform(name, func)
        self.transforms.add(transform)
        return transform
    
    def add_constraint(self, name: str, check_func) -> OrderConstraint:
        """
        Add a new order constraint.
        
        This is how intelligence extends structural constraints.
        """
        constraint = OrderConstraint(name, check_func)
        self.order.add(constraint)
        return constraint
    
    def get_stats(self) -> Dict:
        return {
            "total_particles": self.total_particles,
            "total_dark": self.total_dark,
            "total_domains": len(self.domains),
            "total_heat": float(self.total_heat),
            "experience_count": self.total_experiences,
            "transforms": len(self.transforms.all()),
            "constraints": len(self.order.all()),
            "mode": self.mode.value,
            "uptime": time.time() - self.created_at
        }
    
    def save(self):
        """Persist kernel to disk."""
        if not self.storage_path:
            return
        
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        data = {
            "total_particles": self.total_particles,
            "total_dark": self.total_dark,
            "total_experiences": self.total_experiences,
            "total_heat": float(self.total_heat),
            "created_at": self.created_at,
            "order_history": self.order_history,
            "mode": self.mode.value,
            "domains": {}
        }
        
        for d_id, domain in self.domains.items():
            data["domains"][d_id] = {
                "id": domain.id,
                "name": domain.name,
                "mean_x": domain.mean_x,
                "mean_y": domain.mean_y,
                "var_x": domain.var_x,
                "var_y": domain.var_y,
                "count": domain.count,
                "total_heat": domain.total_heat,
                "created_at": domain.created_at,
                "correlations": domain.correlations,
                "particles": {
                    p_id: {
                        "id": p.id,
                        "heat_k": p.heat.k,
                        "heat_iterations": p.heat.origin_iterations,
                        "polarity_phase": p.polarity.phase,
                        "polarity_mode": p.polarity.mode,
                        "x": p.x,
                        "y": p.y,
                        "transform_used": p.transform_used,
                        "movement_vector": p.movement.vector,
                        "movement_mode": p.movement.mode.value,
                        "movement_chosen": p.movement.chosen,
                        "order_links": p.order_links,
                        "raw_value": str(p.raw_value),
                        "created_at": p.created_at
                    }
                    for p_id, p in domain.particles.items()
                }
            }
        
        with open(self.storage_path / "kernel.json", "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load kernel from disk."""
        if not self.storage_path:
            return
        
        kernel_file = self.storage_path / "kernel.json"
        if not kernel_file.exists():
            return
        
        try:
            with open(kernel_file, "r") as f:
                data = json.load(f)
            
            self.total_particles = data.get("total_particles", 0)
            self.total_dark = data.get("total_dark", 0)
            self.total_experiences = data.get("total_experiences", 0)
            self.total_heat = Heat(k=data.get("total_heat", 0.0))
            self.created_at = data.get("created_at", time.time())
            self.order_history = data.get("order_history", [])
            
            mode_str = data.get("mode", "agency")
            self.mode = MovementMode.AGENCY if mode_str == "agency" else MovementMode.INTELLIGENCE
            
            for d_id, d_data in data.get("domains", {}).items():
                domain = Domain(id=d_data["id"], name=d_data["name"])
                domain.mean_x = d_data.get("mean_x", 0.0)
                domain.mean_y = d_data.get("mean_y", 0.0)
                domain.var_x = d_data.get("var_x", 1.0)
                domain.var_y = d_data.get("var_y", 1.0)
                domain.count = d_data.get("count", 0)
                domain.total_heat = d_data.get("total_heat", 0.0)
                domain.created_at = d_data.get("created_at", time.time())
                domain.correlations = d_data.get("correlations", {})
                
                for p_id, p_data in d_data.get("particles", {}).items():
                    heat = Heat(
                        k=p_data["heat_k"],
                        origin_iterations=p_data.get("heat_iterations", 0)
                    )
                    polarity = Polarity(p_data.get("polarity_phase", 0))
                    polarity.mode = p_data.get("polarity_mode", "agency")
                    
                    move_mode = MovementMode.AGENCY
                    if p_data.get("movement_mode") == "intelligence":
                        move_mode = MovementMode.INTELLIGENCE
                    
                    movement = Movement(
                        vector=tuple(p_data.get("movement_vector", (0, 0))),
                        mode=move_mode,
                        chosen=p_data.get("movement_chosen", False)
                    )
                    
                    particle = Particle(
                        id=p_data["id"],
                        heat=heat,
                        polarity=polarity,
                        x=p_data["x"],
                        y=p_data["y"],
                        transform_used=p_data.get("transform_used", "linear"),
                        movement=movement,
                        order_links=p_data.get("order_links", []),
                        domain_id=d_id,
                        raw_value=p_data.get("raw_value"),
                        created_at=p_data.get("created_at", time.time())
                    )
                    domain.particles[p_id] = particle
                
                self.domains[d_id] = domain
        
        except Exception as e:
            print(f"Error loading kernel: {e}")
