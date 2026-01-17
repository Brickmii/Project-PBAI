"""
Motion Calendar Seed - Identity (OS) Module (v2)
The bounded active model that only updates via the righteousness gate.

Righteousness is a TRANSFORM FUNCTION on k (learnable).
Failed transforms trigger randomization.
Stable transforms become functions.
"""

import json
import time
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field

from core import (
    Particle, Heat, Polarity, Movement, MovementMode,
    Transform, OrderConstraint, DEFAULT_TRANSFORMS
)
from kernel import Domain


@dataclass
class RighteousnessResult:
    """Result of righteousness gate evaluation."""
    passed: bool
    reason: str
    transform_used: Optional[str] = None
    pearson_value: float = 0.0
    counterpart_id: Optional[str] = None


class RighteousnessGate:
    """
    The Righteousness Gate.
    
    Righteousness is not a value - it's a transform operator on k.
    k' = T_θ(k)
    
    The gate:
    1. Checks content redundancy (exact match = reject)
    2. Applies transforms to find righteousness position
    3. Evaluates novelty via correlation
    4. Checks for counterpart (x,y balance)
    
    Failed transforms trigger randomization.
    Stable transforms become functions (learned).
    """
    
    def __init__(self, novelty_threshold: float = 0.3):
        self.novelty_threshold = novelty_threshold
        
        # Pending particles awaiting counterparts
        self.pending: Dict[str, Particle] = {}
        
        # Transform success/failure tracking for learning
        self.transform_results: Dict[str, Dict[str, int]] = {}  # transform -> {success, fail}
        
        # Statistics
        self.total_evaluated = 0
        self.total_passed = 0
        self.total_rejected_redundant = 0
        self.total_rejected_contradiction = 0
        self.total_awaiting = 0
        self.total_transform_failures = 0
    
    def evaluate(self, particle: Particle, identity: 'Identity') -> RighteousnessResult:
        """
        Evaluate if a particle should pass into identity.
        
        Pipeline:
        1. Content check (exact match = redundant)
        2. Transform evaluation (is position valid?)
        3. Novelty check (Pearson correlation)
        4. Counterpart check (x,y balance)
        """
        self.total_evaluated += 1
        
        # Get relevant domain from identity
        domain = identity.get_domain(particle.domain_id)
        
        # === NEW DOMAIN ===
        if domain is None or domain.count == 0:
            self.total_passed += 1
            self._record_transform_success(particle.transform_used)
            return RighteousnessResult(
                passed=True,
                reason="new_domain",
                transform_used=particle.transform_used
            )
        
        # === CONTENT CHECK ===
        content_key = str(particle.raw_value).lower()
        if domain.contains_content(content_key):
            self.total_rejected_redundant += 1
            return RighteousnessResult(
                passed=False,
                reason="redundant",
                transform_used=particle.transform_used,
                pearson_value=1.0
            )
        
        # === TRANSFORM EVALUATION ===
        # Check if the transform produced a valid position
        if not self._validate_transform(particle, domain):
            self.total_transform_failures += 1
            self._record_transform_failure(particle.transform_used)
            # Transform failed - could trigger re-transform with different function
            return RighteousnessResult(
                passed=False,
                reason="transform_failed",
                transform_used=particle.transform_used
            )
        
        # === NOVELTY CHECK ===
        pearson = self._compute_correlation(particle, domain)
        
        if pearson > (1.0 - self.novelty_threshold):
            self.total_rejected_redundant += 1
            return RighteousnessResult(
                passed=False,
                reason="redundant",
                transform_used=particle.transform_used,
                pearson_value=pearson
            )
        
        if pearson < -(1.0 - self.novelty_threshold):
            self.total_rejected_contradiction += 1
            return RighteousnessResult(
                passed=False,
                reason="contradiction",
                transform_used=particle.transform_used,
                pearson_value=pearson
            )
        
        # === COUNTERPART CHECK ===
        if not self._has_counterpart(particle, domain):
            counterpart = self._find_counterpart(particle)
            if counterpart:
                self.total_passed += 1
                self._record_transform_success(particle.transform_used)
                return RighteousnessResult(
                    passed=True,
                    reason="counterpart_found",
                    transform_used=particle.transform_used,
                    pearson_value=pearson,
                    counterpart_id=counterpart.id
                )
            else:
                self.pending[particle.id] = particle
                self.total_awaiting += 1
                return RighteousnessResult(
                    passed=False,
                    reason="awaiting_counterpart",
                    transform_used=particle.transform_used,
                    pearson_value=pearson
                )
        
        # === PASSED ===
        self.total_passed += 1
        self._record_transform_success(particle.transform_used)
        return RighteousnessResult(
            passed=True,
            reason="novel",
            transform_used=particle.transform_used,
            pearson_value=pearson
        )
    
    def _validate_transform(self, particle: Particle, domain: Domain) -> bool:
        """
        Check if transform produced valid righteousness position.
        
        A transform is valid if:
        - Position is within expected bounds
        - Position is not degenerate (NaN, Inf)
        """
        x, y = particle.x, particle.y
        
        # Check for degenerate values
        if math.isnan(x) or math.isnan(y):
            return False
        if math.isinf(x) or math.isinf(y):
            return False
        
        # Check bounds (righteousness should be in [-1, 1] typically)
        if abs(x) > 10 or abs(y) > 10:
            return False
        
        return True
    
    def _compute_correlation(self, particle: Particle, domain: Domain) -> float:
        """Compute novelty correlation with existing domain."""
        if domain.count < 2:
            return 0.0
        
        var_x, var_y = domain.get_variance()
        std_x = max(math.sqrt(var_x), 0.001)
        std_y = max(math.sqrt(var_y), 0.001)
        
        z_x = (particle.x - domain.mean_x) / std_x
        z_y = (particle.y - domain.mean_y) / std_y
        
        distance = math.sqrt(z_x**2 + z_y**2)
        correlation = 1.0 / (1.0 + distance)
        
        return correlation
    
    def _has_counterpart(self, particle: Particle, domain: Domain) -> bool:
        """Check if particle has counterpart in domain."""
        opposite = (particle.quadrant + 2) % 4
        for p in domain.particles.values():
            if p.quadrant == opposite:
                return True
        return False
    
    def _find_counterpart(self, particle: Particle) -> Optional[Particle]:
        """Find counterpart in pending."""
        target = (particle.quadrant + 2) % 4
        for pid, pending in list(self.pending.items()):
            if pending.domain_id == particle.domain_id and pending.quadrant == target:
                del self.pending[pid]
                self.total_awaiting -= 1
                return pending
        return None
    
    def _record_transform_success(self, transform_name: str):
        """Record successful transform application."""
        if transform_name not in self.transform_results:
            self.transform_results[transform_name] = {"success": 0, "fail": 0}
        self.transform_results[transform_name]["success"] += 1
    
    def _record_transform_failure(self, transform_name: str):
        """Record failed transform application."""
        if transform_name not in self.transform_results:
            self.transform_results[transform_name] = {"success": 0, "fail": 0}
        self.transform_results[transform_name]["fail"] += 1
    
    def get_transform_stability(self, transform_name: str) -> float:
        """Get stability of a transform."""
        if transform_name not in self.transform_results:
            return 0.5  # Unknown
        results = self.transform_results[transform_name]
        total = results["success"] + results["fail"]
        if total == 0:
            return 0.5
        return results["success"] / total
    
    def get_stats(self) -> Dict:
        return {
            "total_evaluated": self.total_evaluated,
            "total_passed": self.total_passed,
            "total_rejected_redundant": self.total_rejected_redundant,
            "total_rejected_contradiction": self.total_rejected_contradiction,
            "total_awaiting": self.total_awaiting,
            "total_transform_failures": self.total_transform_failures,
            "pending_count": len(self.pending),
            "pass_rate": self.total_passed / max(self.total_evaluated, 1),
            "transform_results": self.transform_results
        }


class Identity:
    """
    The OS - bounded active model of self.
    
    Only updates via the righteousness gate.
    Contains:
    - Selected features (particles that passed)
    - Learned transforms (righteousness functions)
    - Structural rules (order constraints)
    - Thresholds
    """
    
    def __init__(self, storage_path: Optional[Path] = None):
        self.storage_path = storage_path
        
        # Identity domains
        self.domains: Dict[str, Domain] = {}
        
        # Learned transforms specific to identity
        self.learned_transforms: Dict[str, Transform] = {}
        
        # Learned rules
        self.rules: List[Dict] = []
        
        # Thresholds
        self.thresholds: Dict[str, float] = {
            "activation_minimum": 0.1,
            "correlation_strength": 0.3,
            "transform_stability_required": 0.6
        }
        
        # Statistics
        self.total_integrated = 0
        self.total_heat = Heat(k=0.0)
        self.created_at = time.time()
        
        if storage_path:
            self.load()
    
    def get_domain(self, domain_id: str) -> Optional[Domain]:
        return self.domains.get(domain_id)
    
    def integrate(self, particle: Particle, counterpart: Optional[Particle] = None):
        """Integrate a particle that passed the gate."""
        # Get or create domain
        if particle.domain_id not in self.domains:
            self.domains[particle.domain_id] = Domain(
                id=particle.domain_id,
                name=particle.domain_id
            )
        
        domain = self.domains[particle.domain_id]
        domain.add_particle(particle)
        self.total_integrated += 1
        self.total_heat = self.total_heat.accumulate(particle.heat.k)
        
        # Also integrate counterpart
        if counterpart:
            domain.add_particle(counterpart)
            self.total_integrated += 1
            self.total_heat = self.total_heat.accumulate(counterpart.heat.k)
        
        # Learn rules from order links
        self._learn_rules(particle)
        
        # Update correlations
        self._update_correlations(particle.domain_id)
    
    def learn_transform(self, name: str, func: Callable[[float], Tuple[float, float]]):
        """
        Learn a new transform function.
        
        This is how intelligence acquires new righteousness transforms.
        """
        transform = Transform(name, func)
        self.learned_transforms[name] = transform
        return transform
    
    def get_best_transform(self, k: float) -> Transform:
        """Get best transform for given heat value."""
        # Prefer learned transforms with high stability
        best = None
        best_stability = 0
        
        for t in self.learned_transforms.values():
            if t.stability > best_stability:
                best = t
                best_stability = t.stability
        
        if best and best_stability >= self.thresholds["transform_stability_required"]:
            return best
        
        # Fall back to default
        return DEFAULT_TRANSFORMS[0]
    
    def _learn_rules(self, particle: Particle):
        """Learn order rules from particle's links."""
        for linked_id in particle.order_links:
            linked = self._find_particle(linked_id)
            if linked:
                rule = {
                    "antecedent": linked.domain_id,
                    "consequent": particle.domain_id,
                    "strength": 1.0,
                    "count": 1
                }
                existing = self._find_rule(linked.domain_id, particle.domain_id)
                if existing:
                    existing["strength"] += 0.1
                    existing["count"] += 1
                else:
                    self.rules.append(rule)
    
    def _find_particle(self, particle_id: str) -> Optional[Particle]:
        for domain in self.domains.values():
            if particle_id in domain.particles:
                return domain.particles[particle_id]
        return None
    
    def _find_rule(self, antecedent: str, consequent: str) -> Optional[Dict]:
        for rule in self.rules:
            if rule["antecedent"] == antecedent and rule["consequent"] == consequent:
                return rule
        return None
    
    def _update_correlations(self, domain_id: str):
        if domain_id not in self.domains:
            return
        source = self.domains[domain_id]
        for other_id, other in self.domains.items():
            if other_id == domain_id:
                continue
            if source.count < 2 or other.count < 2:
                continue
            dist = math.sqrt(
                (source.mean_x - other.mean_x)**2 + 
                (source.mean_y - other.mean_y)**2
            )
            corr = 1.0 / (1.0 + dist)
            source.correlations[other_id] = corr
            other.correlations[domain_id] = corr
    
    def get_correlated_domains(self, domain_id: str, min_strength: float = 0.3) -> List[Tuple[str, float]]:
        if domain_id not in self.domains:
            return []
        domain = self.domains[domain_id]
        results = [(k, v) for k, v in domain.correlations.items() if v >= min_strength]
        return sorted(results, key=lambda x: -x[1])
    
    def get_successor_domains(self, domain_id: str) -> List[Tuple[str, float]]:
        successors = []
        for rule in self.rules:
            if rule["antecedent"] == domain_id:
                successors.append((rule["consequent"], rule["strength"]))
        return sorted(successors, key=lambda x: -x[1])
    
    def get_stats(self) -> Dict:
        return {
            "total_integrated": self.total_integrated,
            "total_domains": len(self.domains),
            "total_heat": float(self.total_heat),
            "total_rules": len(self.rules),
            "learned_transforms": len(self.learned_transforms),
            "uptime": time.time() - self.created_at
        }
    
    def save(self):
        if not self.storage_path:
            return
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        data = {
            "total_integrated": self.total_integrated,
            "total_heat": float(self.total_heat),
            "created_at": self.created_at,
            "thresholds": self.thresholds,
            "rules": self.rules,
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
                        "x": p.x,
                        "y": p.y,
                        "transform_used": p.transform_used,
                        "order_links": p.order_links,
                        "raw_value": str(p.raw_value),
                        "created_at": p.created_at
                    }
                    for p_id, p in domain.particles.items()
                }
            }
        
        with open(self.storage_path / "identity.json", "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        if not self.storage_path:
            return
        identity_file = self.storage_path / "identity.json"
        if not identity_file.exists():
            return
        
        try:
            with open(identity_file, "r") as f:
                data = json.load(f)
            
            self.total_integrated = data.get("total_integrated", 0)
            self.total_heat = Heat(k=data.get("total_heat", 0.0))
            self.created_at = data.get("created_at", time.time())
            self.thresholds = data.get("thresholds", self.thresholds)
            self.rules = data.get("rules", [])
            
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
                    heat = Heat(k=p_data.get("heat_k", 1.0))
                    polarity = Polarity()  # Reconstructed
                    movement = Movement.random()
                    
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
            print(f"Error loading identity: {e}")
