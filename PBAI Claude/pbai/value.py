# value.py
from .evaluator import cycle_has_change, cycle_is_logical, cycle_difference
from .pbai_core import State

# Tunable, but fixed for now
NOVELTY_THRESHOLD = 2


def evaluate_cycle(cycle, memory_signatures):
    """
    Determines the value outcome of a cycle.
    Returns: 'pay', 'produce', or 'ignore'
    """

    # 1. Illogical cycles are discarded
    if not cycle_is_logical(cycle):
        return "ignore"

    # 2. No change = no value
    if not cycle_has_change(cycle):
        return "ignore"

    # 3. Measure novelty
    diff = cycle_difference(cycle, memory_signatures)

    # 4. Value decision
    if diff == 0:
        return "ignore"          # already known
    elif diff <= NOVELTY_THRESHOLD:
        return "produce"         # useful refinement
    else:
        return "pay"             # costly novelty