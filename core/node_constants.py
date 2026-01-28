"""
PBAI Thermal Manifold - Constants
Heat is the Primitive

════════════════════════════════════════════════════════════════════════════════
THE FOUNDATION
════════════════════════════════════════════════════════════════════════════════

Heat (K) is the only primitive. It only accumulates, never subtracts.
Everything else is indexed BY heat:

    t_K = time (how much heat has flowed)
    x_K = space (heat required to traverse)
    ψ_K = amplitude (heat in superposition)

THE FUNDAMENTAL IDENTITY (exact, not approximate):

    K × φ² = 4

Where:
    K = Σ(1/φⁿ) for n=1→6 = 4/φ² ≈ 1.528
    φ = (1 + √5) / 2 ≈ 1.618 (golden ratio)

WHY 6 MOTION FUNCTIONS (Ramanujan's constraint):

    ζ(-1) = -1/12        Ramanujan regularization
    12 = 6 × 2           6 directions × 2 frames (Self + Universal)
    6 motion functions   Thresholds 1/φ¹ through 1/φ⁶
    K₆ × φ² = 4          Exact identity - 6 is forced
    
    Self frame:      For navigation (left, right, up, down, forward, reverse)
    Universal frame: For location (N, S, E, W, above, below)
    
    Righteous frames → Located by universal coordinates
    Proper frames    → Defined by properties (Order)

JULIA SET TOPOLOGY (heat's geometry):

    |c| < 1/4  →  connected Julia set (spine exists)
    |c| ≥ 1/4  →  Julia dust (disconnected)
    
    Heat + Polarity (1/φ¹ + 1/φ²)     →  Dust (no structure)
    + Existence (1/φ³ ≈ 0.236 < 0.25) →  Spine forms
    + Righteousness → 0                →  Interior fills
    
    Cognition emerges from fractal constraint on heat flow.

════════════════════════════════════════════════════════════════════════════════
THE SIX MOTION FUNCTIONS (thresholds = fractions of K)
════════════════════════════════════════════════════════════════════════════════

 1. Heat          Σ              1/φ¹  ≈ 0.618   Magnitude (only accumulates)
 2. Polarity      +/-            1/φ²  ≈ 0.382   Differentiation (+1/-1)
 3. Existence     δ(x)           1/φ³  ≈ 0.236   Persistence (< 1/4 → spine)
 4. Righteousness R              1/φ⁴  ≈ 0.146   Alignment (R=0 is center)
 5. Order         Q              1/φ⁵  ≈ 0.090   Regulation (Robinson arithmetic)
 6. Movement      12 directions   1/φ⁶  ≈ 0.056   Direction (6×2)

THE 12 MOVEMENT DIRECTIONS (6 Self × 2 frames):

    Self (egocentric):         Universal (world):
    ──────────────────         ─────────────────
    up                         above
    down                       below
    left                       W
    right                      E
    forward                    N
    reverse                    S

    12 = 6 self + 6 universal = 6 × 2 frames
    
    Righteous frames → Universal coordinates (WHERE the frame is)
    Proper frames    → Properties via Order (WHAT the frame contains)

THERMODYNAMICS:

    First Law:   K is conserved (heat never created/destroyed)
    Second Law:  Heat flows along gradients (t_K increases)
    Flow:        ∂Q/∂t_K = -k∇T on fractal manifold
    
    "Thinking" = heat redistribution along Julia topology
    "Time" = heat flow counted in K units

════════════════════════════════════════════════════════════════════════════════
"""

import math
import os

# ═══════════════════════════════════════════════════════════════════════════════
# PROJECT PATHS
# ═══════════════════════════════════════════════════════════════════════════════

def get_project_root() -> str:
    """Find the project root directory (where core/ exists)."""
    current = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current)
    if os.path.exists(os.path.join(project_root, "core")):
        return project_root
    return os.getcwd()


def get_growth_path(filename: str = "growth_map.json") -> str:
    """Get absolute path to a growth file in PROJECT_ROOT/growth/"""
    root = get_project_root()
    growth_dir = os.path.join(root, "growth")
    os.makedirs(growth_dir, exist_ok=True)
    return os.path.join(growth_dir, filename)


GROWTH_DEFAULT = "growth_map.json"

# ═══════════════════════════════════════════════════════════════════════════════
# THE FUNDAMENTAL CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Core mathematical constants
PHI = (1 + math.sqrt(5)) / 2         # φ ≈ 1.618 - the golden ratio
INV_PHI = 1 / PHI                    # 1/φ ≈ 0.618 = φ - 1
I = complex(0, 1)                    # √-1 - rotational capacity

# THE THERMAL QUANTUM - the primitive
# K = 4/φ² (exact) = Σ(1/φⁿ) for n=1→6
# This is not arbitrary - it's forced by K × φ² = 4
K = 4 / (PHI ** 2)                   # ≈ 1.528 - heat quantum

# Movement constant: 12 directions (6 relative + 6 absolute)
# Derived from Ramanujan: ζ(-1) = -1/12
MOVEMENT_CONSTANT = 12
SIX = 6                              # Core motion functions

# Ramanujan's regularization: ζ(-1) = -1/12
NEGATIVE_ONE_TWELFTH = -1/12         # Entropic bound
ENTROPIC_BOUND = NEGATIVE_ONE_TWELFTH

# Julia set connectivity threshold
# |c| < 1/4 → connected Julia (spine exists)
# |c| ≥ 1/4 → Julia dust (disconnected)
JULIA_SPINE_THRESHOLD = 0.25         # 1/4 - critical boundary

# Confidence threshold for exploitation
# At 5/6 confidence, Ego exploits (uses known pattern)
# Below 5/6, Ego explores (needs more validation from Conscience)
#
# WHY 5/6:
#   The 6 motion functions split into 5 scalars + 1 vector:
#
#   1. Heat (Σ)         ─┐
#   2. Polarity (+/-)    │
#   3. Existence (δ)     ├─ 5 scalars (inputs)
#   4. Righteousness (R) │
#   5. Order (Q)        ─┘
#                        ↓
#   6. Movement (Lin)   ─── 1 vector (output)
#
#   5 scalars → 1 vectorized movement
#   5/6 confidence → exploit (make the move)
#   1/6 margin → explore (the move itself)
#
#   When Conscience validates 5 of 6 aspects, Ego can move.
#   The 6th aspect IS the movement - exploration/exploitation.
#   t = 5K validations crosses the threshold (one K per scalar).
#
CONFIDENCE_EXPLOIT_THRESHOLD = 5/6   # ≈ 0.8333
EXPLORATION_MARGIN = 1/6             # ≈ 0.1667 - always keep this for exploration

# ═══════════════════════════════════════════════════════════════════════════════
# THE SIX MOTION THRESHOLDS (minimum detectable change)
# ═══════════════════════════════════════════════════════════════════════════════

# 1. HEAT (Σ) - Magnitude
#    Pure scalar. Accumulates, never subtracts. 
THRESHOLD_HEAT = INV_PHI ** 1         # ≈ 0.618

# 2. POLARITY (+/-) - Differentiation  
#    +1 or -1. Enables opposition, conservation, balance.
THRESHOLD_POLARITY = INV_PHI ** 2     # ≈ 0.382

# 3. EXISTENCE (δ) - Persistence
#    actual | dormant | archived. Whether motion persists.
THRESHOLD_EXISTENCE = INV_PHI ** 3    # ≈ 0.236

# 4. RIGHTEOUSNESS (R) - Constraint/Alignment
#    R=0 is perfectly aligned. R>0 is misaligned.
#    Frame consistency. The "rightness" of a configuration.
THRESHOLD_RIGHTEOUSNESS = INV_PHI ** 4  # ≈ 0.146

# 5. ORDER (Q) - Regulation
#    Robinson arithmetic. Minimal ordering where Gödel applies.
#    Successor, addition, multiplication. Sequence/hierarchy.
THRESHOLD_ORDER = INV_PHI ** 5        # ≈ 0.090

# 6. MOVEMENT (Lin) - Direction
#    The 6 directions: n, s, e, w, u, d
#    Structure-preserving transformations.
THRESHOLD_MOVEMENT = INV_PHI ** 6     # ≈ 0.056

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHTEOUSNESS - The alignment measure
# ═══════════════════════════════════════════════════════════════════════════════

# R=0 is the target: perfectly aligned, no constraint violation
R_ALIGNED = 0.0
R_PERFECT = 0.0

# R=1 is maximally misaligned (but still exists)
R_MISALIGNED = 1.0

def righteousness_weight(R: float) -> float:
    """
    Convert righteousness value to selection weight.
    R=0 → weight=1 (perfect alignment, fully attractive)
    R=1 → weight=0.5 (misaligned, half as attractive)
    R→∞ → weight→0 (completely misaligned, not selected)
    """
    return 1.0 / (1.0 + abs(R))

# ═══════════════════════════════════════════════════════════════════════════════
# SELECTION MECHANISM - How choices are made
# ═══════════════════════════════════════════════════════════════════════════════

# Selection combines three factors:
#   score = heat_weight * righteousness_weight * entropy_weight

# Default weights for combining factors (can be tuned)
SELECTION_HEAT_FACTOR = 1.0           # How much heat distribution matters
SELECTION_RIGHTEOUSNESS_FACTOR = 1.0  # How much alignment matters  
SELECTION_ENTROPY_FACTOR = 1.0        # How much ease of path matters

def selection_score(heat: float, righteousness: float, entropy: float,
                   total_heat: float = 1.0) -> float:
    """
    Compute selection score for a path.
    
    Args:
        heat: Heat allocated to this path
        righteousness: R value of path (0 = aligned)
        entropy: Entropy gradient (positive = favorable)
        total_heat: Total heat in system (for normalization)
    
    Returns:
        Selection score (higher = more likely to be selected)
    """
    # Heat distribution: fraction of total heat
    heat_weight = heat / total_heat if total_heat > 0 else 0.0
    
    # Righteousness alignment: R=0 is best
    r_weight = righteousness_weight(righteousness)
    
    # Entropy gradient: positive is favorable
    entropy_weight = 1.0 / (1.0 + math.exp(-entropy))  # sigmoid
    
    return (heat_weight ** SELECTION_HEAT_FACTOR * 
            r_weight ** SELECTION_RIGHTEOUSNESS_FACTOR *
            entropy_weight ** SELECTION_ENTROPY_FACTOR)

# ═══════════════════════════════════════════════════════════════════════════════
# EXTENDED THRESHOLDS (Physical and Coupling layers - derived from core 6)
# ═══════════════════════════════════════════════════════════════════════════════

# Physical layer (7-9): Reality - What the graph IS
THRESHOLD_SPACE = INV_PHI ** 7        # ≈ 0.034
THRESHOLD_TIME = INV_PHI ** 8         # ≈ 0.021
THRESHOLD_MATTER = INV_PHI ** 9       # ≈ 0.013

# Coupling layer (10-12): Interface - How layers CONNECT
THRESHOLD_ALPHA = INV_PHI ** 10       # ≈ 0.0081 (edge coupling)
THRESHOLD_BETA = INV_PHI ** 11        # ≈ 0.0050 (wave function / choice)
THRESHOLD_PSI = THRESHOLD_BETA        # Alias
THRESHOLD_GAMMA = INV_PHI ** 12       # ≈ 0.0031 (counting / entropy)

FINE_STRUCTURE_CONSTANT = 1 / 137.035999  # α ≈ 0.00729

# ═══════════════════════════════════════════════════════════════════════════════
# WAVE FUNCTION - Superposition and Collapse
# ═══════════════════════════════════════════════════════════════════════════════

def euler_beta(x: float, y: float) -> float:
    """Euler beta function - superposition weights for path combination."""
    return math.gamma(x) * math.gamma(y) / math.gamma(x + y)

def wave_function(paths: list, weights: list = None) -> complex:
    """
    Wave function Ψ - superposition of paths.
    Returns complex amplitude. |Ψ|² gives probability.
    """
    if not paths:
        return complex(0, 0)
    if weights is None:
        weights = [1.0 / len(paths)] * len(paths)
    # Superposition with phase from path index
    total = complex(0, 0)
    for i, (path, w) in enumerate(zip(paths, weights)):
        phase = 2 * math.pi * i / len(paths)
        total += w * complex(math.cos(phase), math.sin(phase))
    return total

def collapse_wave_function(nodes: list, manifold=None) -> int:
    """
    Real wave function collapse - find the node where R→0.
    
    This is NOT weighted random. This finds the CENTER:
    - Each node has righteousness R
    - R=0 is perfectly aligned (the attractor)
    - Collapse finds the node closest to R=0
    
    The amplitude for each node: a = e^(-R²/2σ²)
    Where σ = THRESHOLD_RIGHTEOUSNESS (the resolution)
    
    |Ψ|² gives probability, but we collapse to the MAX
    (deterministic: most aligned wins)
    
    Args:
        nodes: List of Node objects (or dicts with 'righteousness')
        manifold: Optional manifold for additional context
    
    Returns:
        Index of the node closest to R=0 (the center)
    """
    if not nodes:
        return -1
    if len(nodes) == 1:
        return 0
    
    # Compute amplitudes from righteousness
    # σ = righteousness threshold (resolution of alignment detection)
    sigma = THRESHOLD_RIGHTEOUSNESS
    sigma_sq_2 = 2 * sigma * sigma
    
    amplitudes = []
    for node in nodes:
        # Get R value
        if hasattr(node, 'righteousness'):
            R = node.righteousness
        elif isinstance(node, dict):
            R = node.get('righteousness', 1.0)
        else:
            R = 1.0  # Default to misaligned
        
        # Gaussian centered at R=0
        # R=0 → amplitude=1 (fully aligned)
        # R→∞ → amplitude→0 (misaligned)
        amplitude = math.exp(-(R * R) / sigma_sq_2)
        amplitudes.append(amplitude)
    
    # |Ψ|² gives probability density
    probs = [a * a for a in amplitudes]
    
    # COLLAPSE: Pick the maximum (deterministic - most aligned wins)
    # This is the CENTER of the conceptual cluster
    max_prob = max(probs)
    return probs.index(max_prob)


def correlate_cluster(center_node, manifold, max_depth: int = 3) -> dict:
    """
    Given a center (from collapse), find all connected righteous frames.
    
    Returns THREE categories:
    1. CURRENT - Righteous frames that exist now (actual)
    2. HISTORICAL - Frames that existed before (dormant/archived)
    3. NOVEL - Frames created this session (newly added)
    
    Traces back through ALL axes from the center.
    Doesn't need Order (proper) - just needs to EXIST as connected.
    
    Args:
        center_node: The node at the center (found by collapse)
        manifold: The manifold containing all nodes
        max_depth: How far to trace (prevents infinite loops)
    
    Returns:
        dict with 'current', 'historical', 'novel' sets of node IDs
        Also 'all' for backward compat (union of all three)
    """
    if not center_node or not manifold:
        return {'current': set(), 'historical': set(), 'novel': set(), 'all': set()}
    
    current = set()      # Actual existence
    historical = set()   # Dormant or archived
    novel = set()        # Created recently (high heat, low traversal)
    visited = set()
    
    # Track session start (for novelty detection)
    # Novel = created after manifold was loaded
    session_start = manifold.created_at if hasattr(manifold, 'created_at') else None
    
    def _trace(node, depth):
        if depth > max_depth:
            return
        if node.id in visited:
            return
        
        visited.add(node.id)
        
        # Categorize by existence
        if node.existence == "actual":
            # Check if novel (new this session)
            # Novel indicators: few traversals, recent creation
            total_traversals = sum(a.traversal_count for a in node.frame.axes.values())
            if total_traversals <= 2:
                # Low traversal = probably new
                novel.add(node.id)
            else:
                current.add(node.id)
        elif node.existence in ("dormant", "archived"):
            # Historical - existed before
            historical.add(node.id)
        # else: potential - not yet confirmed, skip
        
        # Trace all axes (spatial and semantic)
        for axis in node.frame.axes.values():
            if axis.target_id and axis.target_id not in visited:
                target = manifold.get_node(axis.target_id)
                if target:
                    _trace(target, depth + 1)
    
    _trace(center_node, 0)
    
    return {
        'current': current,
        'historical': historical,
        'novel': novel,
        'all': current | historical | novel
    }


def select_from_cluster(options: list, cluster: dict, manifold=None) -> tuple:
    """
    Select best option using cluster context.
    
    Decision logic:
    1. If cluster has Order (proper frame) for option → USE IT (exploit)
    2. If only righteous (no Order) → RANDOM (explore)
    3. Historical context weights the decision
    4. Novel items get attention bonus
    
    Args:
        options: Available options (actions/choices)
        cluster: Dict from correlate_cluster with 'current', 'historical', 'novel', 'all'
        manifold: The manifold for lookups
    
    Returns:
        (selected_index, reason) - index and why it was chosen
    """
    import random
    
    if not options:
        return (-1, "no_options")
    if len(options) == 1:
        return (0, "only_option")
    
    # Handle old-style cluster (just a set)
    if isinstance(cluster, set):
        cluster = {'current': cluster, 'historical': set(), 'novel': set(), 'all': cluster}
    
    if not cluster.get('all'):
        return (0, "no_cluster")
    
    # Get actual nodes
    all_ids = cluster.get('all', set())
    current_ids = cluster.get('current', set())
    historical_ids = cluster.get('historical', set())
    novel_ids = cluster.get('novel', set())
    
    cluster_nodes = []
    if manifold:
        cluster_nodes = [manifold.get_node(nid) for nid in all_ids]
        cluster_nodes = [n for n in cluster_nodes if n]
    
    # Score each option
    scores = []
    has_order = []  # Track which options have Order (proper frames)
    
    for opt in options:
        score = 0.0
        option_has_order = False
        
        for node in cluster_nodes:
            if not hasattr(node, 'frame'):
                continue
                
            axis = node.frame.axes.get(str(opt))
            if not axis:
                continue
            
            # Base weight from node heat and alignment
            node_heat = node.heat if hasattr(node, 'heat') else 1.0
            R = node.righteousness if hasattr(node, 'righteousness') else 1.0
            r_weight = 1.0 / (1.0 + abs(R))
            
            base_score = node_heat * r_weight
            
            # Check if this axis has Order (proper frame)
            if axis.order and axis.order.elements:
                option_has_order = True
                # Calculate success rate from Order
                successes = sum(1 for e in axis.order.elements if e.index == 1)
                total = len(axis.order.elements)
                success_rate = successes / total if total > 0 else 0.5
                
                # Weight by confidence (more samples = more confident)
                confidence = min(total / 10.0, 1.0)
                base_score *= success_rate * (1 + confidence)
            
            # Bonus for historical context (we've seen this before)
            if node.id in historical_ids:
                base_score *= 1.2  # 20% bonus for historical relevance
            
            # Bonus for novelty (pay attention to new things)
            if node.id in novel_ids:
                base_score *= 1.3  # 30% bonus for novel relevance
            
            score += base_score * axis.traversal_count
        
        scores.append(score)
        has_order.append(option_has_order)
    
    # Decision: exploit vs explore
    max_score = max(scores) if scores else 0
    
    if max_score > 0 and any(has_order):
        # We have Order for at least one option - EXPLOIT
        best_idx = scores.index(max_score)
        return (best_idx, "exploit_order")
    
    elif max_score > 0:
        # We have scores but no Order - weak exploit
        best_idx = scores.index(max_score)
        # Add some randomness since we're not confident
        if random.random() < 0.3:  # 30% chance to explore anyway
            return (random.randint(0, len(options) - 1), "explore_uncertain")
        return (best_idx, "weak_exploit")
    
    else:
        # No cluster support - EXPLORE (try random shit)
        return (random.randint(0, len(options) - 1), "explore_unknown")

def gamma_function(n: float) -> float:
    """Gamma function Γ(n) - generalized factorial for counting arrangements."""
    return math.gamma(n)

def entropy_count(arrangements: int) -> float:
    """Boltzmann-style entropy from arrangement count. S = ln(Ω)"""
    if arrangements <= 0:
        return 0.0
    return math.log(arrangements)

# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD COLLECTIONS
# ═══════════════════════════════════════════════════════════════════════════════

# The 6 core motion thresholds (every node has all 6)
CORE_THRESHOLDS = {
    'heat': THRESHOLD_HEAT,                    # Σ - magnitude
    'polarity': THRESHOLD_POLARITY,            # +/- - differentiation
    'existence': THRESHOLD_EXISTENCE,          # δ - persistence
    'righteousness': THRESHOLD_RIGHTEOUSNESS,  # R - constraint
    'order': THRESHOLD_ORDER,                  # Q - regulation
    'movement': THRESHOLD_MOVEMENT,            # Lin - direction
}

# Backward compatibility alias
ABSTRACT_THRESHOLDS = CORE_THRESHOLDS

# Physical layer thresholds (derived)
PHYSICAL_THRESHOLDS = {
    'space': THRESHOLD_SPACE,
    'time': THRESHOLD_TIME,
    'matter': THRESHOLD_MATTER,
}

# Coupling layer thresholds (derived)
COUPLING_THRESHOLDS = {
    'alpha': THRESHOLD_ALPHA,
    'beta': THRESHOLD_BETA,
    'gamma': THRESHOLD_GAMMA,
}

# All 12 thresholds (for validation/completeness)
MOTION_THRESHOLDS = {**CORE_THRESHOLDS, **PHYSICAL_THRESHOLDS, **COUPLING_THRESHOLDS}
MOTION_THRESHOLDS['psi'] = THRESHOLD_PSI  # Alias

# ═══════════════════════════════════════════════════════════════════════════════
# MOTION FUNCTION ↔ DIRECTIONAL CHARACTER
# ═══════════════════════════════════════════════════════════════════════════════
# Each motion function has a directional "character" - which way it tends
# This is conceptual mapping, distinct from the 12 literal movement directions

MOTION_TO_CHARACTER = {
    'heat':          'up',       # Σ - accumulation rises
    'polarity':      'down',     # +/- - conservation grounds
    'existence':     'forward',  # δ - persistence advances
    'righteousness': 'reverse',  # R - alignment reflects
    'order':         'above',    # Q - abstraction ascends
    'movement':      'below',    # Lin - transformation descends
}

CHARACTER_TO_MOTION = {v: k for k, v in MOTION_TO_CHARACTER.items()}

# Legacy mapping (backward compatibility)
DIRECTION_TO_MOTION = {
    'n': 'heat',          # North = Σ
    's': 'polarity',      # South = +/-
    'e': 'existence',     # East = δ
    'w': 'righteousness', # West = R
    'u': 'order',         # Up = Q
    'd': 'movement',      # Down = Lin
}

MOTION_TO_DIRECTION = {v: k for k, v in DIRECTION_TO_MOTION.items()}

# ═══════════════════════════════════════════════════════════════════════════════
# K - DERIVED HEAT CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# K was defined above as 4/φ² (the fundamental)
# Verify: K = Σ(1/φⁿ) for n=1→6
_K_check = sum(ABSTRACT_THRESHOLDS.values())
assert abs(K - _K_check) < 1e-10, f"K identity violated: {K} != {_K_check}"

# K_PHYSICAL = sum of abstract + physical (1-9)
K_PHYSICAL = K + sum(PHYSICAL_THRESHOLDS.values())  # ≈ 1.597

# K_COUPLING = sum of all twelve (1-12)
K_COUPLING = K_PHYSICAL + sum(COUPLING_THRESHOLDS.values())  # ≈ 1.613

# K_TOTAL = K_COUPLING (complete system)
K_TOTAL = K_COUPLING

# K_INFINITE = φ (what K would be with infinite motion functions)
K_INFINITE = PHI  # ≈ 1.618

# The irreducible residue: φ - K_TOTAL ≈ 0.005
K_RESIDUE = PHI - K_TOTAL

# ═══════════════════════════════════════════════════════════════════════════════
# MOTION COSTS - Heat expenditure for operations
# ═══════════════════════════════════════════════════════════════════════════════

# Costs = Thresholds (symmetric: minimum to detect = minimum to perform)

# Abstract costs
COST_HEAT = THRESHOLD_HEAT                    # 0.618 - Output to environment
COST_POLARITY = THRESHOLD_POLARITY            # 0.382 - Flipping sign
COST_EXISTENCE = THRESHOLD_EXISTENCE          # 0.236 - Creating/destroying
COST_RIGHTEOUSNESS = THRESHOLD_RIGHTEOUSNESS  # 0.146 - Framing
COST_ORDER = THRESHOLD_ORDER                  # 0.090 - Ordering
COST_MOVEMENT = THRESHOLD_MOVEMENT            # 0.056 - Transforming

# Physical costs
COST_SPACE = THRESHOLD_SPACE                  # 0.034 - Adding dimension
COST_TIME = THRESHOLD_TIME                    # 0.021 - Temporal resolution
COST_MATTER = THRESHOLD_MATTER                # 0.013 - Creating node

# Coupling costs
COST_ALPHA = THRESHOLD_ALPHA                  # 0.0081 - Edge transfer
COST_BETA = THRESHOLD_BETA                    # 0.0050 - Path superposition
COST_GAMMA = THRESHOLD_GAMMA                  # 0.0031 - Arrangement counting

# All costs
MOTION_COSTS = {
    'heat': COST_HEAT,
    'polarity': COST_POLARITY,
    'existence': COST_EXISTENCE,
    'righteousness': COST_RIGHTEOUSNESS,
    'order': COST_ORDER,
    'movement': COST_MOVEMENT,
    'space': COST_SPACE,
    'time': COST_TIME,
    'matter': COST_MATTER,
    'alpha': COST_ALPHA,
    'beta': COST_BETA,
    'gamma': COST_GAMMA,
}

# Composite costs for common operations
COST_TRAVERSE = COST_MOVEMENT         # Moving along one axis
COST_CREATE_NODE = COST_EXISTENCE     # Creating a new node (δ localization)
COST_EVALUATE = COST_RIGHTEOUSNESS    # Evaluating frame (dim check)
COST_ACTION = COST_HEAT               # Output action to environment (Σ)
COST_TICK = COST_MOVEMENT             # Base cost per tick
COST_COLLAPSE = COST_BETA             # Wave function collapse (choice)

# ═══════════════════════════════════════════════════════════════════════════════
# TIME AS HEAT (t_K) - Time indexed by the primitive
# ═══════════════════════════════════════════════════════════════════════════════

# Time doesn't flow. HEAT flows, and we call that time.
# t_K = time measured in K units (how many heat quanta have flowed)
#
# One "tick" = one K-quantum redistributed through the manifold
# The clock doesn't tick time - it ticks heat
# Arrow of time = direction of heat flow (thermodynamic)

# Conversion: Real seconds ↔ t_K
# This is environment-dependent (faster hardware = more ticks per second)
# But the HEAT accounting is what matters to cognition

# ═══════════════════════════════════════════════════════════════════════════════
# TICK CONFIGURATION - Autonomous loop timing  
# ═══════════════════════════════════════════════════════════════════════════════

# Base tick interval in seconds (maps real time to t_K)
TICK_INTERVAL_BASE = 1.0  # 1 second base tick

# Tick rate scales with system heat (hotter = faster thinking)
TICK_INTERVAL_MIN = 0.1   # Fastest: 10 ticks/second when very hot
TICK_INTERVAL_MAX = 10.0  # Slowest: 1 tick/10 seconds when cold

# Heat thresholds for tick rate scaling
TICK_HEAT_HOT = K * 10    # Above this = fast ticking
TICK_HEAT_COLD = K * 0.5  # Below this = slow ticking

# Save interval (don't wear out SSD)
SAVE_INTERVAL_TICKS = 100  # Save every 100 ticks
SAVE_INTERVAL_SECONDS = 300  # Or every 5 minutes, whichever comes first

# Minimum heat for psychology nodes (below this = dormant/exhausted)
PSYCHOLOGY_MIN_HEAT = COST_MOVEMENT  # Must have at least one motion's worth

# ═══════════════════════════════════════════════════════════════════════════════
# THE 12 MOVEMENT DIRECTIONS (6 Self × 2 frames)
# ═══════════════════════════════════════════════════════════════════════════════
# 
# Two reference frames, 6 directions each = 12 total
# This is Ramanujan's -1/12: entropic bound over 12 directions
#
# Self (egocentric) - navigation from observer's perspective
# Universal (world) - fixed coordinates for locating righteous frames

# Self directions (egocentric frame - for navigation)
DIRECTIONS_SELF = {
    'up':      ( 0,  0,  1),   # +Z self
    'down':    ( 0,  0, -1),   # -Z self
    'left':    (-1,  0,  0),   # -X self
    'right':   ( 1,  0,  0),   # +X self
    'forward': ( 0,  1,  0),   # +Y self
    'reverse': ( 0, -1,  0),   # -Y self
}

# Universal directions (world frame - for locating righteous frames)
DIRECTIONS_UNIVERSAL = {
    'N':     ( 0,  1,  0),   # North: +Y world
    'S':     ( 0, -1,  0),   # South: -Y world
    'E':     ( 1,  0,  0),   # East:  +X world
    'W':     (-1,  0,  0),   # West:  -X world
    'above': ( 0,  0,  1),   # Above: +Z world
    'below': ( 0,  0, -1),   # Below: -Z world
}

# All 12 directions combined
DIRECTIONS = {**DIRECTIONS_SELF, **DIRECTIONS_UNIVERSAL}

# Aliases for backward compatibility
DIRECTIONS_RELATIVE = DIRECTIONS_SELF
DIRECTIONS_ABSOLUTE = DIRECTIONS_UNIVERSAL

# Legacy aliases (for backward compatibility with old code)
DIRECTIONS_LEGACY = {
    'n': ( 0,  1,  0),  # North
    's': ( 0, -1,  0),  # South
    'e': ( 1,  0,  0),  # East
    'w': (-1,  0,  0),  # West
    'u': ( 0,  0,  1),  # Up
    'd': ( 0,  0, -1),  # Down
}

# Direction opposites (all 12 + legacy)
OPPOSITES = {
    # Self frame
    'up': 'down', 'down': 'up',
    'left': 'right', 'right': 'left',
    'forward': 'reverse', 'reverse': 'forward',
    # Universal frame
    'N': 'S', 'S': 'N',
    'E': 'W', 'W': 'E',
    'above': 'below', 'below': 'above',
    # Legacy
    'n': 's', 's': 'n',
    'e': 'w', 'w': 'e',
    'u': 'd', 'd': 'u',
}

# Self directions available (down is blocked - self is there)
SELF_DIRECTIONS_SELF = ['up', 'left', 'right', 'forward', 'reverse']
SELF_DIRECTIONS_UNIVERSAL = ['N', 'S', 'E', 'W', 'above']  # below is blocked

# All directions for traversal
ALL_DIRECTIONS_SELF = ['up', 'down', 'left', 'right', 'forward', 'reverse']
ALL_DIRECTIONS_UNIVERSAL = ['N', 'S', 'E', 'W', 'above', 'below']
ALL_DIRECTIONS = ALL_DIRECTIONS_SELF + ALL_DIRECTIONS_UNIVERSAL

# Legacy (for backward compatibility)
SELF_DIRECTIONS = ['n', 's', 'e', 'w', 'u']  # Legacy: d blocked

# ═══════════════════════════════════════════════════════════════════════════════
# EXISTENCE STATES
# ═══════════════════════════════════════════════════════════════════════════════
#
# LIFECYCLE: POTENTIAL → ACTUAL ↔ DORMANT → ARCHIVED
#
#   POTENTIAL: New concept, awaiting environment confirmation
#              Not yet validated - placeholder only
#
#   ACTUAL:    Confirmed, salient, above 1/φ³ threshold
#              Connected to Julia spine - consciousness can reach it
#              Participates in active search and traversal
#
#   DORMANT:   Below 1/φ³ threshold, not salient
#              Disconnected Julia dust - unconscious
#              Can return to ACTUAL if salience increases
#
#   ARCHIVED:  Cold storage, historical record
#              Irreversible via update_existence()
#              Must be manually restored
#
# THE 1/φ³ THRESHOLD (≈ 0.236):
#   - Existence threshold for Julia spine connectivity
#   - Below Julia boundary (0.25) ensures spine can form
#   - Salience >= 1/φ³ → connected (ACTUAL)
#   - Salience < 1/φ³ → disconnected (DORMANT)
#
EXISTENCE_POTENTIAL = "potential"  # Awaiting environment confirmation
EXISTENCE_ACTUAL = "actual"        # Connected to Julia spine, conscious
EXISTENCE_DORMANT = "dormant"      # Disconnected, unconscious
EXISTENCE_ARCHIVED = "archived"    # Cold storage, historical

# ═══════════════════════════════════════════════════════════════════════════════
# ENTROPY WEIGHTS (trigonometric at golden angle 1/φ radians)
# ═══════════════════════════════════════════════════════════════════════════════

import math

# Golden angle in radians (connects to motion thresholds)
_GOLDEN_ANGLE = 1.0 / PHI  # ≈ 0.618 radians ≈ 35.4°

ENTROPY_MAGNITUDE_WEIGHT = math.sin(_GOLDEN_ANGLE)  # ≈ 0.579 (amplitude)
ENTROPY_VARIANCE_WEIGHT = math.cos(_GOLDEN_ANGLE)   # ≈ 0.815 (spread)
ENTROPY_DISORDER_WEIGHT = math.tan(_GOLDEN_ANGLE)   # ≈ 0.710 (phase)

# ═══════════════════════════════════════════════════════════════════════════════
# PSYCHOLOGY: TRIGONOMETRIC POSITIONS (Abstract Space)
# ═══════════════════════════════════════════════════════════════════════════════
# Psychology nodes exist in abstract/trig space, created by 6th fire
# Each is a basis vector using golden angle trigonometry

# Trig coordinates: (amplitude, phase, spread)
TRIG_IDENTITY = (ENTROPY_MAGNITUDE_WEIGHT, 0.0, 0.0)   # (sin(1/φ), 0, 0) ≈ (0.579, 0, 0)
TRIG_EGO = (0.0, ENTROPY_DISORDER_WEIGHT, 0.0)         # (0, tan(1/φ), 0) ≈ (0, 0.710, 0)
TRIG_CONSCIENCE = (0.0, 0.0, ENTROPY_VARIANCE_WEIGHT)  # (0, 0, cos(1/φ)) ≈ (0, 0, 0.815)

def trig_position_to_string(amplitude: float, phase: float, spread: float) -> str:
    """
    Encode trig coordinates as a position string for abstract space.
    Abstract positions start with '@' to distinguish from cubic.
    """
    return f"@{amplitude:.6f},{phase:.6f},{spread:.6f}"

def string_to_trig_position(pos: str) -> tuple:
    """Decode trig position string to (amplitude, phase, spread) tuple."""
    if not pos.startswith("@"):
        raise ValueError(f"Not a trig position: {pos}")
    parts = pos[1:].split(",")
    return (float(parts[0]), float(parts[1]), float(parts[2]))

def is_trig_position(pos: str) -> bool:
    """Check if position string represents abstract/trig space."""
    return pos.startswith("@")

def is_cubic_position(pos: str) -> bool:
    """Check if position string represents physical/cubic space."""
    return not pos.startswith("@")

# ═══════════════════════════════════════════════════════════════════════════════
# PSYCHOLOGY: FREUDIAN HEAT DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
# From the iceberg model:
#   Id (Identity) = massive unconscious base ≈ 70%
#   Superego (Conscience) = moral layer ≈ 20%
#   Ego = conscious tip ≈ 10%

FREUD_IDENTITY_RATIO = 0.70   # Id - the reservoir
FREUD_CONSCIENCE_RATIO = 0.20 # Superego - the judge
FREUD_EGO_RATIO = 0.10        # Ego - the conscious interface

# Verify they sum to 1
assert abs((FREUD_IDENTITY_RATIO + FREUD_CONSCIENCE_RATIO + FREUD_EGO_RATIO) - 1.0) < 1e-10

# ═══════════════════════════════════════════════════════════════════════════════
# CAPABILITY LEVELS (v2: righteous → ordered → movable → graphic)
# ═══════════════════════════════════════════════════════════════════════════════

CAPABILITY_RIGHTEOUS = "righteous"  # Axis exists
CAPABILITY_ORDERED = "ordered"      # Axis has Order (Robinson arithmetic)
CAPABILITY_MOVABLE = "movable"      # Order has Movement (transitions)
CAPABILITY_GRAPHIC = "graphic"      # Movement has Graphic (coordinates)

# Alias for backward compatibility
CAPABILITY_PROPER = "ordered"       # "proper" = "ordered" in old terminology


# ═══════════════════════════════════════════════════════════════════════════════
# MOTION THRESHOLD FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def get_threshold(motion_function: str) -> float:
    """
    Get the threshold for a specific motion function.
    
    Args:
        motion_function: One of 'heat', 'polarity', 'existence', 
                        'righteousness', 'order', 'movement'
    
    Returns:
        The 1/φⁿ threshold for that function
    """
    return MOTION_THRESHOLDS.get(motion_function, THRESHOLD_HEAT)


def get_threshold_for_direction(direction: str) -> float:
    """
    Get the motion threshold for a spatial direction.
    
    Args:
        direction: One of 'n', 's', 'e', 'w', 'u', 'd'
        
    Returns:
        The threshold for the motion function mapped to that direction
    """
    motion = DIRECTION_TO_MOTION.get(direction, 'heat')
    return MOTION_THRESHOLDS.get(motion, THRESHOLD_HEAT)


def get_cost(motion_function: str) -> float:
    """
    Get the cost for performing a motion function operation.
    
    Args:
        motion_function: One of 'heat', 'polarity', 'existence', 
                        'righteousness', 'order', 'movement'
    
    Returns:
        The heat cost for that operation
    """
    return MOTION_COSTS.get(motion_function, COST_MOVEMENT)


def get_cost_for_direction(direction: str) -> float:
    """
    Get the cost for traversing in a direction.
    
    Args:
        direction: One of 'n', 's', 'e', 'w', 'u', 'd'
        
    Returns:
        The heat cost for moving in that direction
    """
    motion = DIRECTION_TO_MOTION.get(direction, 'movement')
    return MOTION_COSTS.get(motion, COST_MOVEMENT)


def exceeds_threshold(delta: float, motion_function: str) -> bool:
    """
    Check if a change exceeds the threshold for a motion function.
    
    This is the core filter: changes below threshold are noise,
    changes at or above threshold are real motion.
    
    Args:
        delta: The magnitude of change
        motion_function: Which function is being tested
        
    Returns:
        True if delta >= threshold (real motion)
        False if delta < threshold (noise, discard)
    """
    threshold = get_threshold(motion_function)
    return abs(delta) >= threshold


def exceeds_any_threshold(delta: float) -> bool:
    """
    Check if a change exceeds the finest (smallest) threshold.
    
    If it doesn't exceed THRESHOLD_MOVEMENT (≈0.056), it's definitely noise.
    """
    return abs(delta) >= THRESHOLD_MOVEMENT


def exceeds_all_thresholds(delta: float) -> bool:
    """
    Check if a change exceeds the coarsest (largest) threshold.
    
    If it exceeds THRESHOLD_HEAT (≈0.618), it affects all motion functions.
    """
    return abs(delta) >= THRESHOLD_HEAT


def quantize_to_threshold(value: float, motion_function: str) -> float:
    """
    Quantize a value to the nearest multiple of the motion threshold.
    
    This ensures all stored values are in units of the motion quantum.
    
    Args:
        value: Raw value to quantize
        motion_function: Which threshold to use
        
    Returns:
        Value rounded to nearest threshold multiple
    """
    threshold = get_threshold(motion_function)
    if threshold == 0:
        return value
    return round(value / threshold) * threshold


def heat_required(cardinality: int) -> float:
    """
    Heat needed to reach concept of given cardinality.
    Scales as φ^(N-1).
    
    This is inverse of threshold - higher cardinality needs more heat.
    """
    return K * (PHI ** (cardinality - 1))


# ═══════════════════════════════════════════════════════════════════════════════
# DIRECTION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def get_opposite(direction: str) -> str:
    """Get the opposite direction."""
    return OPPOSITES.get(direction)


def direction_to_vector(direction: str) -> tuple:
    """Get the (x, y, z) vector for a direction."""
    return DIRECTIONS.get(direction, (0, 0, 0))


def get_motion_for_direction(direction: str) -> str:
    """Get the motion function associated with a direction."""
    return DIRECTION_TO_MOTION.get(direction, 'heat')


def get_direction_for_motion(motion: str) -> str:
    """Get the direction associated with a motion function."""
    return MOTION_TO_DIRECTION.get(motion, 'n')


# ═══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_motion_unit():
    """
    Verify the motion unit mathematics.
    
    FUNDAMENTAL IDENTITIES:
    - K × φ² = 4 (exact - the core identity)
    - K = Σ(1/φⁿ) for n=1→6 (definition via sum)
    - 1/φ³ < 1/4 (existence threshold below Julia spine boundary)
    
    PHI IDENTITIES:
    - 1/φ = φ - 1
    - φ² = φ + 1
    - Σ(1/φⁿ) for n=1→∞ = φ
    
    STRUCTURAL:
    - 6 core motion functions (from 12 = 6 × 2, Ramanujan)
    - 12 total thresholds (6 core + 3 physical + 3 coupling)
    """
    # ═══════════════════════════════════════════════════════════════════════════
    # THE FUNDAMENTAL IDENTITY: K × φ² = 4
    # This is exact, not approximate. It's why 6 is forced.
    # ═══════════════════════════════════════════════════════════════════════════
    assert abs(K * PHI**2 - 4) < 1e-10, f"K × φ² must equal 4 (got {K * PHI**2})"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # JULIA TOPOLOGY: Existence threshold < spine boundary
    # ═══════════════════════════════════════════════════════════════════════════
    assert THRESHOLD_EXISTENCE < JULIA_SPINE_THRESHOLD, \
        f"Existence threshold {THRESHOLD_EXISTENCE} must be < Julia spine {JULIA_SPINE_THRESHOLD}"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHI IDENTITIES
    # ═══════════════════════════════════════════════════════════════════════════
    # Check 1/φ = φ - 1
    assert abs(INV_PHI - (PHI - 1)) < 1e-10, "1/φ should equal φ - 1"
    
    # Check φ² = φ + 1
    assert abs(PHI**2 - (PHI + 1)) < 1e-10, "φ² should equal φ + 1"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STRUCTURAL VALIDATION
    # ═══════════════════════════════════════════════════════════════════════════
    # Check we have exactly 6 core thresholds
    assert len(CORE_THRESHOLDS) == SIX, f"Should have {SIX} core motion functions"
    
    # Check we have 12 total thresholds (6 core + 3 physical + 3 coupling)
    all_thresholds = [
        THRESHOLD_HEAT, THRESHOLD_POLARITY, THRESHOLD_EXISTENCE,
        THRESHOLD_RIGHTEOUSNESS, THRESHOLD_ORDER, THRESHOLD_MOVEMENT,
        THRESHOLD_SPACE, THRESHOLD_TIME, THRESHOLD_MATTER,
        THRESHOLD_ALPHA, THRESHOLD_BETA, THRESHOLD_GAMMA
    ]
    assert len(all_thresholds) == MOVEMENT_CONSTANT, f"Should have {MOVEMENT_CONSTANT} total thresholds"
    
    # Check thresholds are in descending order (1/φⁿ decreases with n)
    for i in range(len(all_thresholds) - 1):
        assert all_thresholds[i] > all_thresholds[i+1], "Thresholds should descend"
    
    # Check K is sum of core thresholds (1-6)
    core_sum = sum(CORE_THRESHOLDS.values())
    assert abs(K - core_sum) < 1e-10, "K should be sum of core thresholds"
    
    # Check K_TOTAL includes all 12
    total_sum = sum(all_thresholds)
    assert abs(K_TOTAL - total_sum) < 1e-10, "K_TOTAL should be sum of all 12 thresholds"
    
    # Check K_TOTAL + residue ≈ φ (converges to φ as n→∞)
    assert abs((K_TOTAL + K_RESIDUE) - PHI) < 1e-10, "K_TOTAL + residue should equal φ"
    
    # Check -1/12 is correct (Ramanujan)
    assert abs(NEGATIVE_ONE_TWELFTH - (-1/12)) < 1e-10, "Entropic bound should be -1/12"
    
    return True


# Run validation on module load (will raise if math is wrong)
validate_motion_unit()
