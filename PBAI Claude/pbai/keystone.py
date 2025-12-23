# keystone.py
# PBAI Keystone: Identity + State Sum only
# NO arithmetic, NO motion, NO shortcuts


# -------------------------
# Identity Channels
# -------------------------

def heat_channel(k):
    # Heat is identity-indexed, not computed
    return {k}


def polarity_channel(k):
    return {"-", "+"}


def existence_channel(k):
    return {0, 1, "t"}


def righteousness_channel(k):
    return {"T", "F", "x", "-x"}


def true_motion_channel(k):
    # Arithmetic grammar lives in pbai_core
    return {"RobinsonArithmetic"}


def movement_channel(k):
    return {
        "←", "→", "↑", "↓",
        "F", "R",
        "N", "S", "E", "W",
        "^", "_"
    }


# -------------------------
# Divisor Mapping
# -------------------------

DIVISOR_CHANNELS = {
    1: heat_channel,
    2: polarity_channel,
    3: existence_channel,
    4: righteousness_channel,
    6: true_motion_channel,
    12: movement_channel,
}


# -------------------------
# State Sum (Keysum)
# -------------------------

def S_k(k):
    return {
        "heat": heat_channel(k),
        "polarity": polarity_channel(k),
        "existence": existence_channel(k),
        "righteousness": righteousness_channel(k),
        "true_motion": true_motion_channel(k),
        "movement": movement_channel(k),
    }


# -------------------------
# Symbolic Divisibility
# -------------------------

def _divides_symbolically(d, k) -> bool:
    """
    Symbolic placeholder for divisor admissibility.

    This function MUST be replaced by:
    - an external admissibility oracle, OR
    - a successor-count-based check using pbai_core

    Keystone does NOT compute divisibility.
    """
    return True  # admissibility delegated upward
