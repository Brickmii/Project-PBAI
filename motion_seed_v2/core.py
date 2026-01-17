"""
Motion Calendar Seed - Core Mathematical Foundations

PBAI Functional Stack:
1. Polarity (±) — infinite opposition
2. Imaginary phase (i) — proto-heat  
3. Existence Gate — Euler's Identity: e^(iπ) + 1 = 0
4. Heat (k) — post-existence real magnitude (no subtraction)
5. Righteousness — learnable transform functions
6. Order — structural constraints (extensible)
7. Movement — vector expression (random or chosen)

Agency randomizes; Intelligence chooses.
"""

import cmath
import math
import random
import hashlib
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Any, Callable
from enum import Enum
import numpy as np


# =============================================================================
# POLARITY - Infinite Opposition
# =============================================================================

class Polarity:
    """
    Polarity is the infinite function.
    Binary, oscillatory, never resolves.
    Supplies: opposing motion, contrast, choice space.
    
    Two modes:
    - Random polarity → agency
    - Chosen polarity → intelligence
    """
    
    def __init__(self, phase: float = None):
        # Phase determines position in infinite oscillation
        # None = random (agency), specific value = chosen (intelligence)
        if phase is None:
            self.phase = random.random() * 2 * math.pi
            self.mode = "agency"
        else:
            self.phase = phase
            self.mode = "intelligence"
    
    @property
    def value(self) -> float:
        """Current polarity value in [-1, +1] range."""
        return math.cos(self.phase)
    
    @property
    def sign(self) -> int:
        """Binary sign: +1 or -1."""
        return 1 if self.value >= 0 else -1
    
    def oscillate(self, delta: float = 0.1) -> 'Polarity':
        """Advance the oscillation (agency) or hold (intelligence can choose)."""
        if self.mode == "agency":
            return Polarity(self.phase + delta * random.gauss(1, 0.3))
        else:
            return self  # Intelligence holds unless it chooses to change
    
    def choose(self, new_phase: float) -> 'Polarity':
        """Intelligence can choose a new polarity state."""
        return Polarity(new_phase)
    
    def oppose(self) -> 'Polarity':
        """Return opposing polarity."""
        return Polarity(self.phase + math.pi)
    
    def __repr__(self):
        return f"Polarity({self.sign:+d}, phase={self.phase:.3f}, mode={self.mode})"


# =============================================================================
# IMAGINARY PHASE - Proto-Heat (Pre-Existence)
# =============================================================================

class ImaginaryPhase:
    """
    Pre-existence state: randomizing √i-like phase magnitude.
    This is proto-heat - not real heat yet.
    
    Heat becomes real only after the existence gate fires.
    """
    
    def __init__(self, polarity: Polarity):
        self.polarity = polarity
        # Start with random complex phase
        self.z = self._initialize_phase()
        self.iterations = 0
        self.max_iterations = 1000
    
    def _initialize_phase(self) -> complex:
        """Initialize with √i-like exploration."""
        # √i = e^(iπ/4) = (1 + i) / √2
        sqrt_i = cmath.exp(1j * math.pi / 4)
        
        # Randomize around √i
        angle_perturbation = random.gauss(0, math.pi / 8)
        magnitude_perturbation = random.gauss(1, 0.2)
        
        base = sqrt_i * magnitude_perturbation
        rotation = cmath.exp(1j * angle_perturbation)
        
        return base * rotation * self.polarity.sign
    
    def iterate(self) -> complex:
        """
        One iteration of phase exploration.
        Searching for existence gate condition.
        """
        self.iterations += 1
        
        # Evolve in complex plane
        # Movement influenced by polarity
        delta_angle = random.gauss(0, 0.1) * self.polarity.value
        delta_mag = random.gauss(0, 0.05)
        
        r, theta = cmath.polar(self.z)
        new_r = max(0.01, r + delta_mag)
        new_theta = theta + delta_angle
        
        self.z = cmath.rect(new_r, new_theta)
        return self.z
    
    def check_existence_gate(self) -> Tuple[bool, float]:
        """
        Check if current state satisfies Euler's Identity.
        
        e^(iπ) + 1 = 0
        
        We check: |e^(iz) + 1| < threshold
        When this is satisfied, existence fires.
        
        Returns: (gate_fired, real_heat)
        """
        # Compute e^(i * z) where z is our complex state
        # But we need to extract real magnitude when gate fires
        
        # The gate condition: when our phase aligns with π
        # such that e^(i*phase) + 1 ≈ 0
        
        _, theta = cmath.polar(self.z)
        
        # How close is our phase to π (or -π)?
        distance_to_pi = min(
            abs(theta - math.pi),
            abs(theta + math.pi),
            abs(theta - math.pi * 3)  # wrap around
        )
        
        # Gate fires when we're close enough to the Euler condition
        threshold = 0.1  # Tunable
        
        if distance_to_pi < threshold:
            # EXISTENCE FIRES
            # Heat becomes the real magnitude
            real_heat = abs(self.z)
            return True, real_heat
        
        return False, 0.0
    
    def run_until_existence(self) -> Tuple[bool, float, int]:
        """
        Run iterations until existence gate fires or max iterations.
        
        Returns: (existed, heat, iterations)
        """
        for _ in range(self.max_iterations):
            self.iterate()
            exists, heat = self.check_existence_gate()
            if exists:
                return True, heat, self.iterations
        
        # Failed to achieve existence
        return False, 0.0, self.iterations


# =============================================================================
# EXISTENCE GATE - Euler's Identity
# =============================================================================

def existence_gate(phase: ImaginaryPhase) -> Tuple[bool, float]:
    """
    The Existence Gate: Euler's Identity.
    
    e^(iπ) + 1 = 0
    
    Separates non-existence (imaginary/phase) from existence (real/magnitude).
    Decides when internal motion becomes measurable.
    
    This is the ONLY place where "before vs after" is meaningful.
    
    Returns: (exists, real_heat)
    """
    return phase.check_existence_gate()


# =============================================================================
# HEAT - Post-Existence Magnitude
# =============================================================================

@dataclass
class Heat:
    """
    Heat (k) - post-existence magnitude.
    
    Properties:
    - No subtraction (irreversible)
    - Does not depend on time
    - Accumulated, one-way growth
    """
    k: float  # The magnitude
    origin_iterations: int = 0  # How many iterations to achieve existence
    created_at: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if self.k < 0:
            raise ValueError("Heat cannot be negative (no subtraction)")
    
    def accumulate(self, additional: float) -> 'Heat':
        """Heat only accumulates, never subtracts."""
        if additional < 0:
            raise ValueError("Cannot subtract from heat")
        return Heat(
            k=self.k + additional,
            origin_iterations=self.origin_iterations,
            created_at=self.created_at
        )
    
    def __add__(self, other: 'Heat') -> 'Heat':
        return Heat(
            k=self.k + other.k,
            origin_iterations=self.origin_iterations,
            created_at=min(self.created_at, other.created_at)
        )
    
    def __float__(self):
        return self.k
    
    def __repr__(self):
        return f"Heat(k={self.k:.4f})"


# =============================================================================
# TRANSFORM - Righteousness Operations
# =============================================================================

class Transform:
    """
    A transform function on heat (k).
    
    Righteousness is not a value - it's a transform operator.
    k' = T_θ(k)
    
    Transforms are learnable.
    Failed transforms trigger randomization.
    Stable transforms become functions.
    """
    
    def __init__(self, name: str, func: Callable[[float], Tuple[float, float]]):
        """
        Args:
            name: Transform identifier
            func: Function that maps k -> (x, y) position
        """
        self.name = name
        self.func = func
        self.use_count = 0
        self.success_count = 0
        self.created_at = time.time()
    
    def apply(self, k: float) -> Tuple[float, float]:
        """Apply transform to heat, get righteousness position."""
        self.use_count += 1
        return self.func(k)
    
    def record_success(self):
        """Record successful use of transform."""
        self.success_count += 1
    
    @property
    def stability(self) -> float:
        """How stable/reliable is this transform."""
        if self.use_count == 0:
            return 0.0
        return self.success_count / self.use_count
    
    def __repr__(self):
        return f"Transform({self.name}, stability={self.stability:.2f})"


# Default transforms
def linear_transform(k: float) -> Tuple[float, float]:
    """Simple linear mapping."""
    x = math.tanh(k - 1)  # Center around k=1
    y = math.tanh(k * 0.5)
    return (x, y)


def log_transform(k: float) -> Tuple[float, float]:
    """Logarithmic mapping (compresses large values)."""
    log_k = math.log(k + 1)
    x = math.tanh(log_k - 1)
    y = math.tanh(log_k * 0.7)
    return (x, y)


def oscillatory_transform(k: float) -> Tuple[float, float]:
    """Oscillatory mapping (periodic structure)."""
    x = math.sin(k * math.pi)
    y = math.cos(k * math.pi * 0.5)
    return (x, y)


DEFAULT_TRANSFORMS = [
    Transform("linear", linear_transform),
    Transform("log", log_transform),
    Transform("oscillatory", oscillatory_transform),
]


# =============================================================================
# ORDER - Structural Constraints
# =============================================================================

@dataclass
class OrderConstraint:
    """
    A structural constraint on transforms.
    
    Order constrains transforms.
    Order can itself be extended.
    Intelligence may:
    - Create new structural constraints
    - Preserve invariants
    - Enforce consistency
    """
    name: str
    check: Callable[[Any, Any], bool]  # (before, after) -> valid?
    created_at: float = field(default_factory=time.time)
    
    def validate(self, before: Any, after: Any) -> bool:
        """Check if transition from before to after is valid."""
        return self.check(before, after)


def succession_constraint(before: Any, after: Any) -> bool:
    """Basic succession: after must follow before in some order."""
    # Default: always valid (no constraint)
    return True


def monotonic_heat_constraint(before: 'Particle', after: 'Particle') -> bool:
    """Heat must not decrease."""
    return float(after.heat) >= float(before.heat)


DEFAULT_CONSTRAINTS = [
    OrderConstraint("succession", succession_constraint),
    OrderConstraint("monotonic_heat", monotonic_heat_constraint),
]


# =============================================================================
# MOVEMENT - Vector Expression
# =============================================================================

class MovementMode(Enum):
    AGENCY = "agency"      # Random movement
    INTELLIGENCE = "intelligence"  # Random or chosen


@dataclass
class Movement:
    """
    Movement is vector expression of k in coordinates.
    Finite, executed, observable.
    
    Two modes:
    - Agency: random movement
    - Intelligence: random OR chosen movement
    """
    vector: Tuple[float, float]
    mode: MovementMode
    chosen: bool = False  # Was this movement chosen (vs random)?
    
    @classmethod
    def random(cls, mode: MovementMode = MovementMode.AGENCY) -> 'Movement':
        """Generate random movement."""
        angle = random.random() * 2 * math.pi
        magnitude = random.gauss(0.5, 0.2)
        magnitude = max(0.1, min(1.0, magnitude))
        
        vector = (
            magnitude * math.cos(angle),
            magnitude * math.sin(angle)
        )
        return cls(vector=vector, mode=mode, chosen=False)
    
    @classmethod
    def choose(cls, direction: Tuple[float, float]) -> 'Movement':
        """Intelligence chooses specific movement."""
        # Normalize
        mag = math.sqrt(direction[0]**2 + direction[1]**2)
        if mag > 0:
            vector = (direction[0]/mag, direction[1]/mag)
        else:
            vector = (0.0, 0.0)
        return cls(vector=vector, mode=MovementMode.INTELLIGENCE, chosen=True)
    
    @property
    def magnitude(self) -> float:
        return math.sqrt(self.vector[0]**2 + self.vector[1]**2)
    
    @property
    def angle(self) -> float:
        return math.atan2(self.vector[1], self.vector[0])


# =============================================================================
# PARTICLE - Complete Motion Unit
# =============================================================================

@dataclass
class Particle:
    """
    A complete motion unit that has passed the existence gate.
    
    Contains all layers:
    - Heat (k): real magnitude
    - Polarity: oscillation state
    - Righteousness position (x, y): from transform
    - Order links: structural connections
    - Movement: vector expression
    """
    id: str
    heat: Heat
    polarity: Polarity
    x: float  # Righteousness position
    y: float
    transform_used: str  # Which transform produced x,y
    movement: Movement
    order_links: List[str] = field(default_factory=list)
    domain_id: str = ""
    raw_value: Any = None
    created_at: float = field(default_factory=time.time)
    
    @classmethod
    def create_from_existence(
        cls,
        heat: Heat,
        polarity: Polarity,
        transform: Transform,
        movement: Movement,
        raw_value: Any = None
    ) -> 'Particle':
        """Create a particle that has passed existence gate."""
        # Apply transform to get righteousness position
        x, y = transform.apply(heat.k)
        
        # Generate ID
        id_content = f"{heat.k}{polarity.phase}{x}{y}{time.time()}"
        particle_id = hashlib.sha256(id_content.encode()).hexdigest()[:16]
        
        return cls(
            id=particle_id,
            heat=heat,
            polarity=polarity,
            x=x,
            y=y,
            transform_used=transform.name,
            movement=movement,
            raw_value=raw_value
        )
    
    @property
    def position(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    @property
    def quadrant(self) -> int:
        """Which righteousness quadrant."""
        if self.x >= 0 and self.y >= 0:
            return 0
        elif self.x < 0 and self.y >= 0:
            return 1
        elif self.x < 0 and self.y < 0:
            return 2
        else:
            return 3


# =============================================================================
# DARK MATTER - Pre-Existence State
# =============================================================================

@dataclass
class DarkParticle:
    """
    A particle that has NOT passed the existence gate.
    
    Has:
    - Polarity (oscillating)
    - Imaginary phase (proto-heat)
    
    Does NOT have:
    - Real heat
    - Righteousness position
    - Order
    - Observable movement
    
    This is dark matter in the Motion Calendar ontology.
    """
    id: str
    polarity: Polarity
    phase: ImaginaryPhase
    iterations_attempted: int = 0
    
    @classmethod
    def create(cls) -> 'DarkParticle':
        """Create a new dark particle."""
        polarity = Polarity()  # Random (agency)
        phase = ImaginaryPhase(polarity)
        
        id_content = f"dark_{polarity.phase}_{time.time()}_{random.random()}"
        particle_id = hashlib.sha256(id_content.encode()).hexdigest()[:16]
        
        return cls(
            id=particle_id,
            polarity=polarity,
            phase=phase
        )
    
    def attempt_existence(self) -> Tuple[bool, Optional[Particle], int]:
        """
        Attempt to cross the existence gate.
        
        Returns: (succeeded, particle_if_succeeded, iterations)
        """
        existed, heat_value, iterations = self.phase.run_until_existence()
        self.iterations_attempted += iterations
        
        if existed:
            heat = Heat(k=heat_value, origin_iterations=self.iterations_attempted)
            # Use default transform
            transform = DEFAULT_TRANSFORMS[0]
            movement = Movement.random()
            
            particle = Particle.create_from_existence(
                heat=heat,
                polarity=self.polarity,
                transform=transform,
                movement=movement
            )
            return True, particle, iterations
        
        return False, None, iterations


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_particle_from_input(
    raw_value: Any,
    mode: MovementMode = MovementMode.AGENCY
) -> Particle:
    """
    Create a particle from raw input.
    
    This is a convenience function that:
    1. Creates imaginary phase from input
    2. Runs existence gate
    3. Applies transform
    4. Returns complete particle
    """
    # Derive initial phase from input
    if isinstance(raw_value, str):
        hash_val = int(hashlib.sha256(raw_value.encode()).hexdigest()[:8], 16)
    else:
        hash_val = hash(str(raw_value))
    
    # Create polarity based on mode
    if mode == MovementMode.AGENCY:
        polarity = Polarity()  # Random
    else:
        # Chosen based on input
        chosen_phase = (hash_val % 1000) / 1000 * 2 * math.pi
        polarity = Polarity(chosen_phase)
    
    # Create imaginary phase
    phase = ImaginaryPhase(polarity)
    
    # Bias the phase based on input for deterministic behavior
    input_bias = (hash_val % 10000) / 10000 * math.pi
    phase.z = cmath.rect(abs(phase.z), input_bias + math.pi * 0.9)
    
    # Run existence gate (should fire quickly due to bias)
    existed, heat_value, iterations = phase.run_until_existence()
    
    if not existed:
        # Force existence with minimal heat
        heat_value = 0.1
        iterations = phase.max_iterations
    
    # Add input-derived heat
    input_heat = 1.0
    if isinstance(raw_value, str):
        input_heat = math.log(len(raw_value) + 1) + 1.0
    elif isinstance(raw_value, (int, float)):
        input_heat = math.log(abs(raw_value) + 1) + 1.0
    
    heat = Heat(k=heat_value + input_heat, origin_iterations=iterations)
    
    # Select transform (could be learned later)
    transform = DEFAULT_TRANSFORMS[0]
    
    # Create movement
    if mode == MovementMode.AGENCY:
        movement = Movement.random(mode)
    else:
        # Intelligence could choose direction based on context
        movement = Movement.random(mode)  # For now, still random
    
    return Particle.create_from_existence(
        heat=heat,
        polarity=polarity,
        transform=transform,
        movement=movement,
        raw_value=raw_value
    )
