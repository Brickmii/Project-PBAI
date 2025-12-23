# api.py - Complete PBAI API Interface

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .translator import translate, get_completed_cycles
from .evaluator import evaluate_cycle, cycle_signature
from .pbai_core import State



@dataclass
class PBAIResponse:
    """Structured response from PBAI system"""
    state: State
    completed_cycles: List[Dict]
    decisions: List[str]
    pressure: int
    memory_count: int

    def to_dict(self) -> Dict:
        return {
            "state": {
                "heat": list(self.state.heat),
                "polarity": list(self.state.polarity),
                "existence": list(self.state.existence),
                "righteousness": list(self.state.righteousness),
                "motion": str(self.state.motion),
                "movement": self.state.movement,
                "phase": self.state.phase,
                "pressure": self.state.pressure,
            },
            "completed_cycles": self.completed_cycles,
            "decisions": self.decisions,
            "pressure": self.pressure,
            "memory_count": self.memory_count,
        }


class PBAI:
    """
    Main PBAI interface - stateful wrapper around PBAI core
    """

    def __init__(self, identity: str):
        self.identity = identity
        self.current_state: Optional[State] = None
        self.tick_count = 0

    def step(self, external: Dict[str, Any]) -> PBAIResponse:
        """
        Process one tick of external input through PBAI system.

        Args:
            external: {
                "movement": str (one of '↑↓←→_' or hardware symbol),
                "sensors": Optional[Dict],
                "timestamp": Optional[Any],
            }

        Returns:
            PBAIResponse with state, cycles, decisions, and metadata
        """
        movement = external.get("movement", "_")

        # 1. Translate external world -> internal state
        self.current_state = translate(self.identity, movement)
        self.tick_count += 1

        # 2. Get any completed cycles
        completed_cycles = get_completed_cycles()

        # 3. Evaluate all completed cycles
        decisions = []
        for cycle in completed_cycles:
            decision = evaluate_cycle(cycle, self.current_state.memory_signatures)
            decisions.append(decision)

            # Store meaningful cycles
            if decision != "ignore":
                sig = cycle_signature(cycle)
                self.current_state.memory_signatures.append(sig)

        # 4. Update pressure based on decisions
        pay_count = decisions.count("pay")
        produce_count = decisions.count("produce")
        self.current_state.pressure += pay_count - (produce_count // 2)

        return PBAIResponse(
            state=self.current_state,
            completed_cycles=completed_cycles,
            decisions=decisions,
            pressure=self.current_state.pressure,
            memory_count=len(self.current_state.memory_signatures),
        )

    def get_state(self) -> Optional[State]:
        """Get current internal state"""
        return self.current_state

    def reset(self):
        """Reset PBAI to initial state"""
        self.current_state = None
        self.tick_count = 0
        # Note: translator state needs reset too (handled separately)


# Convenience function for single-step usage
def pbai_step(identity: str, external: Dict[str, Any]) -> Dict:
    """
    Stateless convenience function - creates new PBAI instance each call.
    For stateful usage, use PBAI class directly.

    Args:
        identity: Agent identifier
        external: External input dict with 'movement' key

    Returns:
        Dictionary representation of PBAIResponse
    """
    pbai = PBAI(identity)
    response = pbai.step(external)
    return response.to_dict()


# Decision-to-Action mapping utilities
def decision_to_exploration_bias(decisions: List[str]) -> float:
    """
    Convert PBAI decisions to exploration bias for action selection.

    Returns:
        float in [0, 1] where:
        - 0.0 = fully exploit (stick with known patterns)
        - 1.0 = fully explore (try new things)
    """
    if not decisions:
        return 0.5  # neutral

    pay_count = decisions.count("pay")
    produce_count = decisions.count("produce")
    ignore_count = decisions.count("ignore")

    # More 'pay' decisions = need more exploration
    # More 'produce' decisions = exploitation is working
    # More 'ignore' decisions = neutral/need change

    exploration = (pay_count + ignore_count * 0.5) / len(decisions)
    return min(max(exploration, 0.0), 1.0)


def should_change_strategy(decisions: List[str], pressure: int) -> bool:
    """
    Determine if agent should change strategy based on decisions and pressure.

    Returns:
        True if strategy change recommended
    """
    if pressure > 5:
        return True  # High pressure = change needed

    if not decisions:
        return False

    # Too many 'pay' decisions without 'produce'
    pay_ratio = decisions.count("pay") / len(decisions)
    return pay_ratio > 0.7