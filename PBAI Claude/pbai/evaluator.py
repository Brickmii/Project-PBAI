# evaluator.py
from .pbai_core import compute_internal_delta

NEUTRAL = "_"
PHASE_CYCLE = (0, 1, 2, 3, 4, 5)
VALID_MOVEMENTS = {"_", "↑", "↓", "←", "→", "F", "R", "N", "S", "E", "W", "^"}


def evaluate_cycle(cycle, memory_signatures):
    """
    Decide what to do with a completed cycle.
    Returns: 'pay', 'produce', or 'ignore'
    """

    if not cycle:
        return "ignore"

    movements = tuple(step["movement"] for step in cycle)

    # novelty check
    known = any(
        sig["movements"] == movements
        for sig in memory_signatures
    )

    if known:
        return "ignore"

    # basic change + logic heuristics (expand later)
    has_change = any(m != "_" for m in movements)

    if has_change:
        return "pay"

    return "produce"


def cycle_has_change(cycle):
    """
    Returns True if any tick in the cycle represents non-neutral movement.
    """
    for tick in cycle:
        if tick["movement"] != NEUTRAL:
            return True
    return False


def cycle_is_logical(cycle):
    """
    Returns True if the cycle is structurally consistent.
    """

    # 1. Phase completeness
    phases = [tick["phase"] for tick in cycle]
    if tuple(phases) != PHASE_CYCLE:
        return False

    # 2. Motion monotonicity
    motions = [tick["motion"] for tick in cycle]
    for i in range(1, len(motions)):
        if motions[i] == motions[i - 1]:
            return False

    # 3. Movement symbol validity
    for tick in cycle:
        if tick["movement"] not in VALID_MOVEMENTS:
            return False

    return True


def cycle_signature(cycle):
    """
    Returns a structural signature of a cycle for comparison.
    """
    movements = tuple(tick["movement"] for tick in cycle)
    activity_count = sum(1 for m in movements if m != "_")
    return {
        "movements": movements,
        "activity": activity_count,
    }


def cycle_difference(cycle, memory_signatures):
    """
    Computes how different this cycle is from previously seen cycles.
    Returns a non-negative integer difference score.
    """

    sig = cycle_signature(cycle)

    if not memory_signatures:
        return len(sig["movements"])  # maximally novel

    differences = []

    for past in memory_signatures:
        diff = 0

        # Movement sequence difference (Hamming distance)
        for a, b in zip(sig["movements"], past["movements"]):
            if a != b:
                diff += 1

        # Activity difference
        diff += abs(sig["activity"] - past["activity"])

        differences.append(diff)

    return min(differences)