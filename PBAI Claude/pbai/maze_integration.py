# maze_integration.py - Connect PBAI to maze environment

from typing import Dict, Tuple, Optional
import random
from pbai.api import PBAI, decision_to_exploration_bias, should_change_strategy


ACTION_TO_MOVEMENT = {
    "up": "↑",
    "down": "↓",
    "left": "←",
    "right": "→",
}

MOVEMENT_TO_ACTION = {v: k for k, v in ACTION_TO_MOVEMENT.items()}


class MazeWithPBAI:
    """
    Wrapper that connects PBAI cognitive system to maze environment.
    """

    def __init__(self, maze_env, agent_id: str = "maze_agent_1"):
        self.maze = maze_env
        self.agent_id = agent_id
        self.pbai = PBAI(agent_id)

        # Track action history for decision-making
        self.action_history = []
        self.stuck_counter = 0
        self.last_position = None
        self.last_decisions = []
        self.last_pressure = 0
        self.pos_history = []  # recent positions
        self.pos_visits = {}  # {(r,c): count}

    def step(self, action: str) -> Dict:
        """
        Execute one action in maze, update PBAI, return comprehensive state.

        Args:
            action: One of "up", "down", "left", "right"

        Returns:
            Dict with PBAI state, maze state, and decision info
        """
        # Attempt move in maze
        success = self.maze.move(action)

        # Convert action to PBAI movement symbol
        movement = ACTION_TO_MOVEMENT.get(action, "_")
        if not success:
            movement = "_"  # Failed move = no movement

        # Update PBAI internal state
        pbai_response = self.pbai.step({"movement": movement})
        self.last_decisions = pbai_response.decisions
        self.last_pressure = pbai_response.pressure

        # Track if stuck
        current_pos = self.maze.player_pos
        self.pos_visits[current_pos] = self.pos_visits.get(current_pos, 0) + 1
        if current_pos == self.last_position:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
        self.last_position = current_pos
        # track recent positions (for loop detection)
        self.pos_history.append(current_pos)
        if len(self.pos_history) > 60:
            self.pos_history.pop(0)


        # keep latest decisions/pressure available to policy
        self.last_decisions = pbai_response.decisions
        self.last_pressure = pbai_response.pressure

        self.action_history.append(action)

        self.action_history.append(action)

        # Path-based ignore: revisiting same position enough times
        path_ignore = (self.pos_visits.get(current_pos, 0) >= 3)

        decisions = list(pbai_response.decisions)
        if path_ignore:
            decisions.append("ignore")

        return {
            "pbai_state": pbai_response.state,
            "pbai_decisions": decisions,  # <- return the modified decisions
            "pbai_pressure": pbai_response.pressure,
            "movement_symbol": movement,
            "move_success": success,
            "at_goal": self.maze.at_goal(),
            "position": current_pos,
            "stuck_counter": self.stuck_counter,
        }

    def get_pbai_suggested_action(self) -> str:
        """
        Improved policy:
        - Avoid immediate backtracking oscillations (A-B-A-B).
        - At intersections, prefer exits not taken recently.
        - Use lightweight visit-count scoring (encourage novel directions).
        - Increase exploration when pressure/stuck is high.
        """
        if self.pbai.current_state is None:
            return random.choice(["up", "down", "left", "right"])

        valid_moves = self.maze.get_valid_moves()
        if not valid_moves:
            return "up"

        # --- Track per-position move counts (create on demand) ---
        if not hasattr(self, "_pos_move_counts"):
            self._pos_move_counts = {}  # {(r,c): {action: count}}

        pos = self.maze.player_pos
        if pos not in self._pos_move_counts:
            self._pos_move_counts[pos] = {m: 0 for m in ["up", "down", "left", "right"]}

        # --- Identify immediate backtrack option (reverse last action) ---
        reverse = {"up": "down", "down": "up", "left": "right", "right": "left"}
        last = self.action_history[-1] if self.action_history else None
        backtrack = reverse.get(last) if last else None

        # --- Detect A-B-A-B oscillation pattern and penalize it hard ---
        oscillating = (
                len(self.action_history) >= 4 and
                self.action_history[-1] == self.action_history[-3] and
                self.action_history[-2] == self.action_history[-4]
        )

        # Combine (weighted)
        base_bias = self._calculate_exploration_bias()
        decision_bias = decision_to_exploration_bias(getattr(self, "last_decisions", []))

        # loopiness: if current position appears often in recent history, crank exploration
        pos = self.maze.player_pos
        recent = getattr(self, "pos_history", [])
        loop_hits = recent.count(pos)
        loop_bias = min(loop_hits / 6.0, 1.0)  # 0..1

        exploration_bias = max(base_bias, decision_bias, loop_bias)

        # --- Intersection detection: if 3+ valid moves, treat as junction ---
        is_junction = len(valid_moves) >= 3

        # --- Recent memory: avoid repeating the same turn choices ---
        recent = self.action_history[-6:] if len(self.action_history) >= 6 else self.action_history

        # --- Score actions ---
        scored = []
        for a in valid_moves:
            score = 0.0

            # Prefer actions we haven't taken much from this position
            count = self._pos_move_counts[pos].get(a, 0)
            score += 1.0 / (1.0 + count)  # higher when count is low

            # At junctions, strongly prefer exits not used recently
            if is_junction and a not in recent:
                score += 0.75

            # Penalize immediate backtrack unless it's the only option
            if backtrack and a == backtrack and len(valid_moves) > 1:
                score -= 0.9

            # Penalize oscillation pattern
            if oscillating and backtrack and a == backtrack:
                score -= 2.0

            # Add exploration noise proportional to exploration_bias
            score += random.random() * (0.25 + 0.75 * exploration_bias)

            scored.append((score, a))

        scored.sort(reverse=True, key=lambda x: x[0])
        chosen = scored[0][1]

        # Record chosen action count for this position
        self._pos_move_counts[pos][chosen] = self._pos_move_counts[pos].get(chosen, 0) + 1

        return chosen

    def _calculate_exploration_bias(self) -> float:
        """Calculate how much agent should explore vs exploit"""
        if self.pbai.current_state is None:
            return 0.5

        # High pressure = need to explore more
        pressure = self.pbai.current_state.pressure
        pressure_bias = min(pressure / 10.0, 1.0)

        # Stuck = need to explore
        stuck_bias = min(self.stuck_counter / 5.0, 1.0)

        return max(pressure_bias, stuck_bias)

    def reset(self):
        """Reset both maze and PBAI"""
        self.maze.reset()
        self.pbai.reset()
        self.action_history.clear()
        self.stuck_counter = 0
        self.last_position = None


# Standalone function for simple integration
def run_maze_with_pbai(maze_env, max_steps: int = 1000) -> Dict:
    """
    Run maze with PBAI making autonomous decisions.

    Args:
        maze_env: Maze environment instance
        max_steps: Maximum steps before giving up

    Returns:
        Dict with final statistics
    """
    wrapper = MazeWithPBAI(maze_env)

    steps = 0
    total_pressure = 0
    decisions_log = []

    while steps < max_steps and not wrapper.maze.at_goal():
        # Let PBAI suggest action
        action = wrapper.get_pbai_suggested_action()

        # Execute step
        result = wrapper.step(action)

        steps += 1
        total_pressure += result["pbai_pressure"]
        decisions_log.extend(result["pbai_decisions"])

        # Optional: render every N steps
        if steps % 10 == 0:
            print(f"Step {steps}: Pressure={result['pbai_pressure']}, "
                  f"Decisions={result['pbai_decisions']}")

    success = wrapper.maze.at_goal()

    return {
        "success": success,
        "steps": steps,
        "total_pressure": total_pressure,
        "final_position": wrapper.maze.player_pos,
        "decisions_summary": {
            "pay": decisions_log.count("pay"),
            "produce": decisions_log.count("produce"),
            "ignore": decisions_log.count("ignore"),
        }
    }