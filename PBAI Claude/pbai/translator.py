# translator.py - Fixed version with reset capability
from .keystone import S_k
from .pbai_core import State, advance_motion, advance_phase

TIME_TICK = "_"
CYCLE_LENGTH = 6

# Global state (module-level)
_last_motion = None
_last_phase = None
_cycle_buffer = []
_completed_cycles = []


def reset_translator():
    """Reset translator to initial state"""
    global _last_motion, _last_phase, _cycle_buffer, _completed_cycles
    _last_motion = None
    _last_phase = None
    _cycle_buffer.clear()
    _completed_cycles.clear()


def translate(k, movement="_"):
    """
    Translate external identity and movement into PBAI State.

    Args:
        k: Identity key
        movement: Movement symbol (default "_" for neutral)

    Returns:
        State object with current internal representation
    """
    global _last_motion, _last_phase

    # Get channel data from keystone
    sk = S_k(k)

    # Advance internal motion
    _last_motion = advance_motion(_last_motion)
    _last_phase = advance_phase(_last_phase)

    # Create tick snapshot
    tick_snapshot = {
        "motion": _last_motion,
        "phase": _last_phase,
        "movement": movement,
    }

    # Buffer management
    _cycle_buffer.append(tick_snapshot)

    # Complete cycle when buffer full
    if len(_cycle_buffer) == CYCLE_LENGTH:
        _completed_cycles.append(_cycle_buffer.copy())
        _cycle_buffer.clear()

    # Return current state
    return State(
        heat=sk.get("heat"),
        polarity=sk.get("polarity"),
        existence=sk.get("existence"),
        righteousness=sk.get("righteousness"),
        motion=_last_motion,
        movement=movement,
        phase=_last_phase,
    )


def get_completed_cycles():
    """
    Get all completed cycles and clear the buffer.

    Returns:
        List of completed cycles (each cycle is list of 6 tick snapshots)
    """
    global _completed_cycles
    cycles = _completed_cycles.copy()
    _completed_cycles.clear()
    return cycles


def get_current_cycle_progress():
    """
    Get current incomplete cycle progress.

    Returns:
        Tuple of (buffer_length, snapshots)
    """
    return len(_cycle_buffer), _cycle_buffer.copy()


if __name__ == "__main__":
    # Test translator
    print("Testing translator...")

    s1 = translate("test_identity", "↑")
    print(f"Tick 1: motion={s1.motion}, phase={s1.phase}, movement={s1.movement}")

    s2 = translate("test_identity", "→")
    print(f"Tick 2: motion={s2.motion}, phase={s2.phase}, movement={s2.movement}")

    # Test cycle completion
    for i in range(4):
        s = translate("test_identity", "_")
        print(f"Tick {i + 3}: motion={s.motion}, phase={s.phase}")

    cycles = get_completed_cycles()
    print(f"\nCompleted cycles: {len(cycles)}")

    if cycles:
        print(f"First cycle movements: {[t['movement'] for t in cycles[0]]}")