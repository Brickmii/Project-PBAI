from dataclasses import dataclass
from typing import Any, Dict, Tuple
MOTION_CYCLE = (0, 1, 2, 3, 4, 5)

# -------------------------
# Robinson Arithmetic Core
# -------------------------

class Zero:
    def __repr__(self):
        return "0"


ZERO = Zero()


@dataclass(frozen=True)
class Successor:
    prev: Any

    def __repr__(self):
        return f"S({self.prev})"


def is_zero(x) -> bool:
    return x is ZERO


def successor(x):
    # Axiom: S(x) != 0
    return Successor(x)


def equal(x, y) -> bool:
    # Structural equality only (no numeric collapse)
    if x is y:
        return True
    if isinstance(x, Successor) and isinstance(y, Successor):
        return equal(x.prev, y.prev)
    return False


def add(x, y):
    # x + 0 = x
    if is_zero(y):
        return x
    # x + S(y) = S(x + y)
    if isinstance(y, Successor):
        return successor(add(x, y.prev))
    raise TypeError("Invalid addition operand")


def mul(x, y):
    # x * 0 = 0
    if is_zero(y):
        return ZERO
    # x * S(y) = (x * y) + x
    if isinstance(y, Successor):
        return add(mul(x, y.prev), x)
    raise TypeError("Invalid multiplication operand")


# -------------------------
# State & Motion
# -------------------------

@dataclass
class State:
    def __init__(self, heat, polarity, existence, righteousness, motion, movement, phase=None):
        self.heat = heat
        self.polarity = polarity
        self.existence = existence
        self.righteousness = righteousness
        self.motion = motion
        self.movement = movement
        self.phase = phase
        self.function_memory = []  # ordered history of changes
        self.pattern_memory = {}  # NEW

        # NEW
        self.pressure = 0
        self.memory_signatures = []


    def __repr__(self):
        return (
            f"State("
            f"heat={self.heat}, "
            f"polarity={self.polarity}, "
            f"existence={self.existence}, "
            f"righteousness={self.righteousness}, "
            f"motion={self.motion}, "
            f"movement={self.movement}, "
            f"phase={self.phase}"
            f")"
        )



    def signature(self) -> Tuple:
        return (
            self.heat,
            self.polarity,
            self.existence,
            self.righteousness,
            self.motion,
            self.movement,
        )


# -------------------------
# State Sum (Keystone)
# -------------------------

class StateSum:
    """
    StateSum does NOT collapse states.
    Equality requires full channel equivalence.
    """

    def __init__(self):
        self.states: Dict[Tuple, State] = {}

    def insert(self, state: State):
        sig = state.signature()
        self.states[sig] = state

    def equal(self, a: State, b: State) -> bool:
        return all(
            equal(x, y) if isinstance(x, Successor) else x == y
            for x, y in zip(a.signature(), b.signature())
        )

    def __len__(self):
        return len(self.states)


# -------------------------
# Sanity Guards
# -------------------------

def forbid_induction():
    raise RuntimeError(
        "Induction is forbidden in PBAI. "
        "Construct successors explicitly."
    )
def advance_motion(current_motion):
    """
    Advance internal motion history by one step.
    """
    if current_motion is None:
        return Successor(ZERO)
    return Successor(current_motion)

def advance_phase(current_phase):
    """
    Advance cyclic motion phase (period 6).
    """
    if current_phase is None:
        return MOTION_CYCLE[0]

    idx = MOTION_CYCLE.index(current_phase)
    return MOTION_CYCLE[(idx + 1) % len(MOTION_CYCLE)]
def snapshot_state(state: State) -> Dict[str, Any]:
    return {
        "heat": set(state.heat),
        "polarity": set(state.polarity),
        "existence": set(state.existence),
        "righteousness": set(state.righteousness),
        "motion": state.motion,
        "movement": state.movement,
        "phase": state.phase,
        "pressure": state.pressure,
    }


def compute_internal_delta(prev: dict, curr: dict) -> dict:
    return {
        k: (prev.get(k), curr.get(k))
        for k in curr
        if prev.get(k) != curr.get(k)
    }



def update_memory(state: State, prev_snapshot: Dict, curr_snapshot: Dict):
    delta = compute_internal_delta(prev_snapshot, curr_snapshot)

    if not delta:
        return None

    state.function_memory.append(delta)
    pattern = tuple(sorted(delta.keys()))
    state.pattern_memory[pattern] = state.pattern_memory.get(pattern, 0) + 1

    return delta

def step_state(state: State):
    prev = snapshot_state(state)

    state.motion = advance_motion(state.motion)
    state.phase = advance_phase(state.phase)

    curr = snapshot_state(state)

    return update_memory(state, prev, curr)
