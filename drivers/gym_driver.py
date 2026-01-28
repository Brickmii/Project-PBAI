"""
PBAI Gym Driver - Unified driver for all Gymnasium environments

Routes to specialized game handlers based on environment type:
- Grid games (FrozenLake, CliffWalking, Taxi): spatial reasoning with cells
- Card games (Blackjack): situation-based decision tracking  
- Continuous (CartPole, MountainCar): adaptive baseline learning

Architecture:
    GymDriver (this file)
        ├── Encoder/Decoder (gym_adapters/) - I/O translation
        ├── GameHandler (internal) - game-specific learning
        └── DriverNode (core/) - persistent knowledge
        ↓
    Gym Environment
"""

import logging
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from time import time
from abc import ABC, abstractmethod

try:
    import gymnasium as gym
    GYM_AVAILABLE = True
except ImportError:
    gym = None
    GYM_AVAILABLE = False

from .environment import Driver, Port, Perception, Action, ActionResult, PortState
from .gym_adapters.encoders import create_encoder, ObservationEncoder
from .gym_adapters.decoders import create_decoder, ActionDecoder

from core.driver_node import DriverNode, SensorReport, MotorAction, MotorType, ActionPlan, press
from core.nodes import Node, Axis, Order, Element
from core import K

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# GYM PORT
# ═══════════════════════════════════════════════════════════════════════════════

class GymPort(Port):
    """Port for Gymnasium environments."""
    
    def __init__(self, env_name: str, **env_kwargs):
        super().__init__(port_id=f"gym_{env_name}")
        self.env_name = env_name
        self.env_kwargs = env_kwargs
        self.env = None
    
    def connect(self) -> bool:
        if not GYM_AVAILABLE:
            logger.error("gymnasium not installed")
            return False
        try:
            self.env = gym.make(self.env_name, **self.env_kwargs)
            self.state = PortState.CONNECTED
            return True
        except Exception as e:
            logger.error(f"Failed to create Gym environment: {e}")
            self.state = PortState.ERROR
            return False
    
    def disconnect(self) -> bool:
        if self.env:
            self.env.close()
            self.env = None
        self.state = PortState.DISCONNECTED
        return True
    
    def send(self, message) -> bool:
        return True
    
    def receive(self, timeout: float = 1.0):
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# GYM STATE
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class GymState:
    """Current state of the gym environment."""
    observation: Any = None
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: Dict = field(default_factory=dict)
    step_count: int = 0
    episode_count: int = 0
    episode_reward: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# GAME HANDLERS - Specialized learning for different game types
# ═══════════════════════════════════════════════════════════════════════════════

class GameHandler(ABC):
    """Base class for game-specific learning handlers."""
    
    def __init__(self, env_name: str, manifold):
        self.env_name = env_name
        self.manifold = manifold
        self.task_node: Optional[Node] = None
    
    def ensure_task_node(self) -> Node:
        """Ensure task node exists (called during driver activation)."""
        concept = self.env_name.split("-")[0].lower()
        return self._get_or_create_task_node(concept)
    
    @abstractmethod
    def get_state_key(self, observation, encoder) -> str:
        """Generate semantic state key."""
        pass
    
    @abstractmethod
    def record_outcome(self, state_key: str, action: str, reward: float, done: bool):
        """Record outcome for learning."""
        pass
    
    @abstractmethod
    def get_action_score(self, state_key: str, action: str) -> float:
        """Score an action based on learned outcomes."""
        pass
    
    def _get_or_create_task_node(self, concept: str) -> Node:
        """Get or create task frame node."""
        if self.task_node:
            return self.task_node
        
        node = self.manifold.get_node_by_concept(concept)
        if node:
            self.task_node = node
            return node
        
        self.task_node = Node(
            concept=concept,
            position="u",
            heat=K,
            polarity=1,
            existence="actual",
            righteousness=1.0,
            order=1
        )
        self.manifold.add_node(self.task_node)
        return self.task_node
    
    def _get_or_create_state_node(self, state_key: str) -> Node:
        """Get or create state node within task frame."""
        node = self.manifold.get_node_by_concept(state_key)
        if node:
            return node
        
        # Create inside task frame
        task = self._get_or_create_task_node(self.env_name.split("-")[0].lower())
        
        # Find position
        existing = len([n for n in self.manifold.nodes.values() 
                       if n.position.startswith(task.position + "d")])
        position = f"{task.position}d{existing}"
        
        node = Node(
            concept=state_key,
            position=position,
            heat=K,
            polarity=1,
            existence="actual",
            righteousness=0.5,
            order=2
        )
        self.manifold.add_node(node)
        
        # Connect task → state
        task.add_axis(state_key, node.id)
        
        return node
    
    def _get_or_create_action_axis(self, state_node: Node, action: str) -> Axis:
        """Get or create axis for state→action."""
        axis_name = f"{state_node.concept}_{action}"
        axis = state_node.get_axis(axis_name)
        
        if not axis:
            axis = state_node.add_axis(axis_name, f"outcome_{axis_name}")
            axis.make_proper()
        
        if not axis.order:
            axis.make_proper()
        
        return axis


class GridGameHandler(GameHandler):
    """
    Handler for grid-based games: FrozenLake, CliffWalking, Taxi.
    
    PLAN-BASED APPROACH:
    1. On episode start: Read full map via SensorReport
    2. Plan path from start to goal (BFS avoiding holes)
    3. Execute plan step by step
    4. If slip detected: replan from current position
    5. Record plan success/failure at episode end
    
    This is how a human plays - look at the board, plan a route, execute.
    """
    
    # Action name to direction delta (row, col)
    ACTION_DELTAS = {
        # FrozenLake/CliffWalking
        "left": (0, -1), "right": (0, 1), "up": (-1, 0), "down": (1, 0),
        # Taxi
        "south": (1, 0), "north": (-1, 0), "east": (0, 1), "west": (0, -1),
    }
    
    # Reverse: delta to action name
    DELTA_TO_ACTION = {
        (0, -1): "left", (0, 1): "right", (-1, 0): "up", (1, 0): "down"
    }
    
    def __init__(self, env_name: str, manifold, gym_env=None):
        super().__init__(env_name, manifold)
        self.gym_env = gym_env
        self.grid_map = None
        self.n_rows = 4
        self.n_cols = 4
        self.goal_pos = None
        self.start_pos = (0, 0)
        self.holes = set()
        
        # Plan tracking
        self.current_plan: Optional[ActionPlan] = None
        self.plan_step: int = 0
        self.expected_pos: Tuple[int, int] = (0, 0)
        self.plan_start_time: float = 0.0
        
        # Configure grid size based on env
        if "8x8" in env_name:
            self.n_cols = self.n_rows = 8
        elif "CliffWalking" in env_name:
            self.n_cols, self.n_rows = 12, 4
        elif "Taxi" in env_name:
            self.n_cols = self.n_rows = 5
        
        # Extract map if available
        self._extract_map()
    
    def _extract_map(self):
        """Extract map layout from gym environment."""
        if not self.gym_env:
            return
        
        unwrapped = self.gym_env.unwrapped
        
        # Try FrozenLake-style desc first
        if hasattr(unwrapped, 'desc') and unwrapped.desc is not None:
            self._extract_from_desc(unwrapped.desc)
            return
        
        # Try CliffWalking (no desc, but known structure)
        if "CliffWalking" in self.env_name:
            self._extract_cliffwalking()
            return
        
        # Try Taxi
        if "Taxi" in self.env_name:
            self._extract_taxi()
            return
        
        logger.warning(f"Could not extract map for {self.env_name}")
    
    def _extract_from_desc(self, desc):
        """Extract from FrozenLake-style desc attribute."""
        self.n_rows = len(desc)
        self.n_cols = len(desc[0]) if self.n_rows > 0 else 4
        
        self.grid_map = []
        self.holes = set()
        
        for r, row in enumerate(desc):
            row_chars = []
            for c, cell in enumerate(row):
                char = chr(cell) if isinstance(cell, int) else cell.decode()
                char = char.upper()
                row_chars.append(char)
                
                if char == 'G':
                    self.goal_pos = (r, c)
                elif char == 'S':
                    self.start_pos = (r, c)
                elif char == 'H':
                    self.holes.add((r, c))
            
            self.grid_map.append(row_chars)
        
        if not self.goal_pos:
            self.goal_pos = (self.n_rows - 1, self.n_cols - 1)
        
        logger.info(f"Map (desc): {self.n_rows}x{self.n_cols}, goal={self.goal_pos}, "
                   f"holes={len(self.holes)}, start={self.start_pos}")
    
    def _extract_cliffwalking(self):
        """
        Extract CliffWalking map (no desc attribute, but known structure).
        
        CliffWalking is always 4x12:
        - Start: bottom-left (3, 0)
        - Goal: bottom-right (3, 11)  
        - Cliff: bottom row between start and goal (3, 1-10)
        - Safe: everywhere else
        
        The optimal path goes UP from start, RIGHT across the top, then DOWN to goal.
        """
        self.n_rows = 4
        self.n_cols = 12
        self.start_pos = (3, 0)   # Bottom-left
        self.goal_pos = (3, 11)   # Bottom-right
        
        # Build map: bottom row (except start/goal) is cliff
        self.grid_map = []
        self.holes = set()  # Cliff cells treated as holes
        
        for r in range(self.n_rows):
            row_chars = []
            for c in range(self.n_cols):
                if r == 3 and c == 0:
                    row_chars.append('S')  # Start
                elif r == 3 and c == 11:
                    row_chars.append('G')  # Goal
                elif r == 3 and 0 < c < 11:
                    row_chars.append('H')  # Cliff (treated as hole)
                    self.holes.add((r, c))
                else:
                    row_chars.append('F')  # Safe floor
            self.grid_map.append(row_chars)
        
        logger.info(f"Map (CliffWalking): {self.n_rows}x{self.n_cols}, "
                   f"start={self.start_pos}, goal={self.goal_pos}, "
                   f"cliff_cells={len(self.holes)}")
    
    def _extract_taxi(self):
        """
        Extract Taxi map (5x5 grid with fixed wall structure).
        
        Taxi has dynamic pickup/dropoff locations encoded in observation.
        """
        self.n_rows = 5
        self.n_cols = 5
        self.start_pos = (0, 0)  # Variable in practice
        self.goal_pos = None     # Dynamic - depends on dropoff location
        
        # Taxi has a fixed 5x5 grid, no holes
        self.grid_map = [['F'] * 5 for _ in range(5)]
        self.holes = set()
        
        logger.info(f"Map (Taxi): {self.n_rows}x{self.n_cols}")
    
    def get_position(self, observation) -> Tuple[int, int]:
        """Get row, col from observation."""
        try:
            state = int(observation)
            row = state // self.n_cols
            col = state % self.n_cols
            return (row, col)
        except:
            return (0, 0)
    
    def _get_cell(self, row: int, col: int) -> str:
        """Get what's at a cell."""
        if row < 0 or row >= self.n_rows or col < 0 or col >= self.n_cols:
            return 'W'  # Wall
        if self.grid_map:
            return self.grid_map[row][col]
        return 'F'
    
    def _is_safe(self, row: int, col: int) -> bool:
        """Check if a cell is safe to traverse."""
        if row < 0 or row >= self.n_rows or col < 0 or col >= self.n_cols:
            return False
        cell = self._get_cell(row, col)
        return cell in ('F', 'S', 'G')  # Frozen, Start, Goal are safe
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SENSOR REPORT - Full map awareness
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_map_report(self) -> SensorReport:
        """
        Create a SensorReport with full map information.
        Called at episode start so PBAI can plan.
        """
        objects = []
        
        # Add all map features as objects
        if self.grid_map:
            for r, row in enumerate(self.grid_map):
                for c, cell in enumerate(row):
                    obj = {
                        "type": cell,
                        "pos": (r, c),
                        "row": r,
                        "col": c
                    }
                    if cell == 'H':
                        obj["danger"] = True
                    elif cell == 'G':
                        obj["goal"] = True
                    elif cell == 'S':
                        obj["start"] = True
                    objects.append(obj)
        
        # Build description
        desc = f"{self.env_name} {self.n_rows}x{self.n_cols} grid. "
        desc += f"Start: {self.start_pos}. Goal: {self.goal_pos}. "
        desc += f"Holes: {len(self.holes)} at {list(self.holes)[:5]}..."
        
        report = SensorReport.vision(
            description=desc,
            objects=objects,
            threats=[{"type": "hole", "pos": h} for h in self.holes]
        )
        report.measurements = {
            "rows": float(self.n_rows),
            "cols": float(self.n_cols),
            "holes": float(len(self.holes)),
            "goal_row": float(self.goal_pos[0]) if self.goal_pos else 0.0,
            "goal_col": float(self.goal_pos[1]) if self.goal_pos else 0.0,
        }
        
        return report
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PATH PLANNING - BFS to find safe route
    # ═══════════════════════════════════════════════════════════════════════════
    
    def plan_path(self, start: Tuple[int, int], goal: Tuple[int, int] = None) -> Optional[ActionPlan]:
        """
        Plan a path from start to goal using BFS.
        Avoids holes. Returns ActionPlan with steps.
        """
        if goal is None:
            goal = self.goal_pos
        if goal is None:
            return None
        
        from collections import deque
        
        # BFS
        queue = deque([(start, [])])  # (position, path)
        visited = {start}
        
        while queue:
            (row, col), path = queue.popleft()
            
            # Reached goal?
            if (row, col) == goal:
                # Convert path to ActionPlan
                steps = []
                for action_name in path:
                    motor = MotorAction(
                        motor_type=MotorType.KEY_PRESS,
                        key=action_name,
                        name=action_name,
                        heat_cost=1.0
                    )
                    steps.append(motor)
                
                plan_name = f"path_{start[0]}{start[1]}_to_{goal[0]}{goal[1]}"
                plan = ActionPlan(
                    name=plan_name,
                    goal=f"reach_{goal}",
                    steps=steps,
                    heat_cost=float(len(steps))
                )
                
                logger.info(f"Planned path: {len(steps)} steps from {start} to {goal}")
                return plan
            
            # Explore neighbors
            for (dr, dc), action_name in self.DELTA_TO_ACTION.items():
                nr, nc = row + dr, col + dc
                
                if (nr, nc) not in visited and self._is_safe(nr, nc):
                    visited.add((nr, nc))
                    queue.append(((nr, nc), path + [action_name]))
        
        logger.warning(f"No path found from {start} to {goal}")
        return None
    
    def start_episode(self, observation) -> Optional[ActionPlan]:
        """
        Called at episode start. Creates initial plan.
        """
        start = self.get_position(observation)
        self.expected_pos = start
        self.plan_step = 0
        self.plan_start_time = time()
        
        # Plan path
        self.current_plan = self.plan_path(start, self.goal_pos)
        
        if self.current_plan:
            logger.info(f"Episode plan: {len(self.current_plan.steps)} steps")
        
        return self.current_plan
    
    def get_next_action(self, observation) -> Optional[str]:
        """
        Get next action from current plan.
        Handles slip detection and replanning.
        """
        if not self.current_plan or not self.current_plan.steps:
            return None
        
        current_pos = self.get_position(observation)
        
        # Slip detection: are we where we expected?
        if current_pos != self.expected_pos:
            logger.info(f"Slip detected: expected {self.expected_pos}, at {current_pos}")
            
            # Replan from current position
            self.current_plan = self.plan_path(current_pos, self.goal_pos)
            self.plan_step = 0
            
            if not self.current_plan:
                return None
        
        # Get next step
        if self.plan_step >= len(self.current_plan.steps):
            return None  # Plan complete
        
        action = self.current_plan.steps[self.plan_step]
        action_name = action.name or action.key
        
        # Update expected position
        if action_name in self.ACTION_DELTAS:
            dr, dc = self.ACTION_DELTAS[action_name]
            self.expected_pos = (current_pos[0] + dr, current_pos[1] + dc)
        
        self.plan_step += 1
        return action_name
    
    def end_episode(self, success: bool):
        """
        Called at episode end. Records plan outcome.
        """
        if self.current_plan:
            duration = time() - self.plan_start_time
            self.current_plan.record_execution(success, duration)
            
            # Store plan in manifold for learning
            plan_key = f"plan_{self.current_plan.name}"
            self._record_plan_outcome(plan_key, success)
            
            logger.info(f"Plan {'SUCCESS' if success else 'FAIL'}: "
                       f"{self.current_plan.name} ({self.current_plan.success_rate:.0%})")
        
        self.current_plan = None
        self.plan_step = 0
    
    def _record_plan_outcome(self, plan_key: str, success: bool):
        """Record plan outcome in manifold for learning."""
        if not self.manifold:
            return
        
        plan_node = self._get_or_create_state_node(plan_key)
        axis = self._get_or_create_action_axis(plan_node, "execute")
        
        axis.order.elements.append(Element(
            node_id=f"{plan_key}_exec_{len(axis.order.elements)}",
            index=1 if success else 0
        ))
        
        if success:
            plan_node.add_heat(K)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEGACY METHODS (for compatibility)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_state_key(self, observation, encoder) -> str:
        """Get semantic state key for current position."""
        pos = self.get_position(observation)
        row, col = pos
        
        # Goal direction
        if self.goal_pos:
            dr = self.goal_pos[0] - row
            dc = self.goal_pos[1] - col
            goal_dir = ""
            if dr > 0: goal_dir += f"{dr}d"
            elif dr < 0: goal_dir += f"{-dr}u"
            if dc > 0: goal_dir += f"{dc}r"
            elif dc < 0: goal_dir += f"{-dc}l"
            if not goal_dir: goal_dir = "G"
        else:
            goal_dir = "?"
        
        # Adjacent cells
        adj = {
            'u': self._get_cell(row - 1, col),
            'd': self._get_cell(row + 1, col),
            'l': self._get_cell(row, col - 1),
            'r': self._get_cell(row, col + 1),
        }
        adj_str = f"{adj['u']}u{adj['d']}d{adj['l']}l{adj['r']}r"
        
        prefix = "fl" if "FrozenLake" in self.env_name else "cw" if "CliffWalking" in self.env_name else "gr"
        return f"{prefix}_{goal_dir}_{adj_str}"
    
    def record_outcome(self, state_key: str, action: str, reward: float, done: bool):
        """Record individual step outcome (for fallback learning)."""
        if not done:
            return
        
        # Record as before for fallback single-step learning
        state_node = self._get_or_create_state_node(state_key)
        axis = self._get_or_create_action_axis(state_node, action)
        
        if "CliffWalking" in self.env_name:
            outcome = 0 if reward < -50 else 1
        else:
            outcome = 1 if reward > 0 else 0
        
        axis.order.elements.append(Element(
            node_id=f"{state_key}_{action}_{len(axis.order.elements)}",
            index=outcome
        ))
        
        if outcome == 1:
            state_node.add_heat(K)
    
    def get_action_score(self, state_key: str, action: str) -> float:
        """Score action (used when no plan available)."""
        state_node = self.manifold.get_node_by_concept(state_key) if self.manifold else None
        if not state_node:
            return 0.5
        
        axis = state_node.get_axis(f"{state_key}_{action}")
        if not axis or not axis.order or not axis.order.elements:
            return 0.5
        
        goods = sum(1 for e in axis.order.elements if e.index == 1)
        return goods / len(axis.order.elements)


class BlackjackGameHandler(GameHandler):
    """
    Handler for Blackjack.
    
    Tracks situation → action → win/loss with semantic keys like "h16v10".
    """
    
    def get_state_key(self, observation, encoder) -> str:
        """Convert observation to blackjack situation key."""
        if observation is None:
            return f"{self.env_name}_unknown"
        
        try:
            player_sum, dealer_card, usable_ace = observation
            soft = "s" if usable_ace else "h"
            return f"bj_{soft}{player_sum}v{dealer_card}"
        except:
            return encoder.encode_key(observation) if encoder else f"{self.env_name}_unknown"
    
    def record_outcome(self, state_key: str, action: str, reward: float, done: bool):
        """Record outcome - only care about final result."""
        if not done:
            return  # Blackjack only has terminal rewards
        
        state_node = self._get_or_create_state_node(state_key)
        axis = self._get_or_create_action_axis(state_node, action)
        
        # Win = 1, Loss = 0, Push = skip
        if reward > 0:
            outcome = 1
            logger.info(f"Blackjack WIN: {state_key} + {action}")
        elif reward < 0:
            outcome = 0
            logger.info(f"Blackjack LOSS: {state_key} + {action}")
        else:
            return  # Push - don't record
        
        axis.order.elements.append(Element(
            node_id=f"{state_key}_{action}_{len(axis.order.elements)}",
            index=outcome
        ))
        axis.traversal_count += 1
        
        # Heat for wins
        if outcome == 1:
            state_node.add_heat(K * 0.5)
    
    def get_action_score(self, state_key: str, action: str) -> float:
        """Score action based on win rate."""
        state_node = self.manifold.get_node_by_concept(state_key)
        if not state_node:
            return 0.5
        
        axis_name = f"{state_key}_{action}"
        axis = state_node.get_axis(axis_name)
        
        if not axis or not axis.order or not axis.order.elements:
            return 0.5
        
        wins = sum(1 for e in axis.order.elements if e.index == 1)
        total = len(axis.order.elements)
        
        return wins / total if total > 0 else 0.5


class GenericGameHandler(GameHandler):
    """
    Handler for continuous control games: CartPole, MountainCar, etc.
    
    Uses adaptive baseline for success determination.
    """
    
    def __init__(self, env_name: str, manifold):
        super().__init__(env_name, manifold)
        self._baseline = 0.0
        self._alpha = 0.1
    
    def get_state_key(self, observation, encoder) -> str:
        """Use encoder for state key."""
        if encoder:
            return encoder.encode_key(observation)
        return f"{self.env_name}_unknown"
    
    def record_outcome(self, state_key: str, action: str, reward: float, done: bool):
        """Record with adaptive baseline."""
        state_node = self._get_or_create_state_node(state_key)
        axis = self._get_or_create_action_axis(state_node, action)
        
        # Success relative to baseline
        success = reward > self._baseline
        self._baseline = (1 - self._alpha) * self._baseline + self._alpha * reward
        
        axis.order.elements.append(Element(
            node_id=f"{state_key}_{action}_{len(axis.order.elements)}",
            index=1 if success else 0
        ))
        axis.traversal_count += 1
    
    def get_action_score(self, state_key: str, action: str) -> float:
        """Score based on success rate."""
        state_node = self.manifold.get_node_by_concept(state_key)
        if not state_node:
            return 0.5
        
        axis_name = f"{state_key}_{action}"
        axis = state_node.get_axis(axis_name)
        
        if not axis or not axis.order or not axis.order.elements:
            return 0.5
        
        successes = sum(1 for e in axis.order.elements if e.index == 1)
        total = len(axis.order.elements)
        
        return successes / total if total > 0 else 0.5


# ═══════════════════════════════════════════════════════════════════════════════
# GYM DRIVER
# ═══════════════════════════════════════════════════════════════════════════════

class GymDriver(Driver):
    """
    Unified Gymnasium Driver - routes to specialized game handlers.
    
    Game type detection:
    - "FrozenLake", "CliffWalking", "Taxi" → GridGameHandler (PLAN-BASED)
    - "Blackjack" → BlackjackGameHandler
    - else → GenericGameHandler
    
    For grid games, uses PLAN-BASED approach:
    1. On episode start: get full map, plan path to goal
    2. Execute plan step by step
    3. Handle slips by replanning
    4. Record plan outcomes for learning
    """
    
    DRIVER_ID = "gym"
    DRIVER_NAME = "Gymnasium Driver"
    DRIVER_VERSION = "4.0.0"  # Plan-based version
    SUPPORTED_ACTIONS = ["move", "interact", "observe", "wait"]
    HEAT_SCALE = 1.0
    
    # Environment classifications
    GRID_ENVS = ["FrozenLake", "CliffWalking", "Taxi"]
    CARD_ENVS = ["Blackjack"]
    SPARSE_ENVS = ["FrozenLake", "CliffWalking", "Taxi", "Blackjack"]
    
    def __init__(self, env_name: str, manifold=None, **env_kwargs):
        port = GymPort(env_name, **env_kwargs)
        super().__init__(port, manifold=manifold)
        
        self.env_name = env_name
        self.DRIVER_ID = f"gym_{env_name}"
        self.DRIVER_NAME = f"Gym: {env_name}"
        
        # Encoder/Decoder
        self.encoder: Optional[ObservationEncoder] = None
        self.decoder: Optional[ActionDecoder] = None
        
        # State
        self.state = GymState()
        self._action_map: Dict[str, int] = {}
        
        # Game handler - set after initialize
        self.game_handler: Optional[GameHandler] = None
        
        # Current state key (for outcome recording)
        self._current_state_key: str = ""
        
        # DriverNode for persistence
        self.driver_node = None
        
        # Plan-based control for grid games
        self._use_planning = any(grid in env_name for grid in self.GRID_ENVS)
        self._episode_just_started = True
        self._plan_initialized = False
    
    @property
    def gym_env(self):
        return self.port.env if isinstance(self.port, GymPort) else None
    
    def _create_game_handler(self) -> GameHandler:
        """Create appropriate game handler based on env type."""
        if any(grid in self.env_name for grid in self.GRID_ENVS):
            logger.info(f"Using GridGameHandler for {self.env_name}")
            return GridGameHandler(self.env_name, self.manifold, gym_env=self.gym_env)
        elif any(card in self.env_name for card in self.CARD_ENVS):
            logger.info(f"Using BlackjackGameHandler for {self.env_name}")
            return BlackjackGameHandler(self.env_name, self.manifold)
        else:
            logger.info(f"Using GenericGameHandler for {self.env_name}")
            return GenericGameHandler(self.env_name, self.manifold)
    
    def initialize(self) -> bool:
        """Initialize driver and connect to gym environment."""
        if not self.port.connect():
            return False
        
        env = self.gym_env
        if not env:
            return False
        
        # Create encoder/decoder
        self.encoder = create_encoder(env.observation_space, self.env_name)
        self.decoder = create_decoder(env.action_space, self.env_name)
        
        # Build action map
        self.SUPPORTED_ACTIONS = self.decoder.action_names.copy()
        for i, name in enumerate(self.decoder.action_names):
            self._action_map[name] = i
        
        # Create game handler
        if self.manifold:
            self.game_handler = self._create_game_handler()
            # Ensure task node exists for heat isolation
            self.game_handler.ensure_task_node()
        
        # Create DriverNode for persistence
        if self.manifold:
            clean_name = self.env_name.replace("-v0", "").replace("-v1", "").lower()
            self.driver_node = DriverNode(clean_name, self.manifold)
            self._register_motors()
        else:
            self.driver_node = None
        
        # Initial reset
        obs, info = env.reset()
        self.state = GymState(observation=obs, info=info)
        
        logger.info(f"GymDriver initialized: {self.env_name}, actions={self.decoder.action_names}")
        return True
    
    def _register_motors(self):
        """Register available actions as motor patterns AND create motor nodes."""
        if not self.driver_node or not self.decoder:
            return
        
        for action_name in self.decoder.action_names:
            # Register motor with DriverNode
            motor = MotorAction(motor_type=MotorType.KEY_PRESS, key=action_name)
            self.driver_node.register_motor(action_name, motor)
            
            # Create a node for this motor action in manifold
            if self.manifold:
                motor_concept = f"{action_name}_motor"
                motor_node = self.manifold.get_node_by_concept(motor_concept)
                if not motor_node:
                    motor_node = Node(
                        concept=motor_concept,
                        position="un",  # Under driver
                        heat=K * 0.5,
                        polarity=1,
                        existence="actual",
                        righteousness=0.5
                    )
                    motor_node.add_tag("motor")
                    motor_node.add_tag(f"action:{action_name}")
                    self.manifold.add_node(motor_node)
                    logger.debug(f"Created motor node: {motor_concept}")
    
    def heat_planned_action(self, action_name: str):
        """
        ADD HEAT to the planned action's motor node.
        
        This integrates planning with PBAI's heat-based decision system.
        The clock will propagate this heat, and DecisionNode will select
        based on which motor has the most heat.
        """
        if not self.manifold:
            return
        
        motor_concept = f"{action_name}_motor"
        motor_node = self.manifold.get_node_by_concept(motor_concept)
        
        if motor_node:
            # Add significant heat to make this action preferred
            motor_node.add_heat(K * 2)  # Double K to stand out
            logger.debug(f"Heated motor {action_name}: heat={motor_node.heat:.2f}")
        else:
            logger.warning(f"Motor node not found: {motor_concept}")
    
    def _decay_motor_heat(self):
        """
        Decay all motor node heat after action execution.
        
        This ensures motor heat doesn't accumulate across steps.
        Each step starts fresh with the planned action getting heat.
        """
        if not self.manifold:
            return
        
        for action in self.get_available_actions():
            motor_concept = f"{action}_motor"
            motor_node = self.manifold.get_node_by_concept(motor_concept)
            if motor_node and motor_node.heat > K * 0.5:
                # Decay toward baseline
                motor_node.heat = max(K * 0.5, motor_node.heat * 0.5)
    
    def shutdown(self) -> bool:
        """Shutdown driver."""
        if self.driver_node:
            self.driver_node.save()
        return self.port.disconnect()
    
    def perceive(self) -> Perception:
        """Convert gym observation to Perception."""
        if not self.encoder or self.state.observation is None:
            return Perception(
                source_driver=self.DRIVER_ID,
                properties={"state_key": f"{self.env_name}_no_obs"}
            )
        
        # Get state key from game handler or encoder
        if self.game_handler:
            state_key = self.game_handler.get_state_key(self.state.observation, self.encoder)
        else:
            state_key = self.encoder.encode_key(self.state.observation)
        
        # Validate state key
        if state_key is None or "None" in str(state_key):
            state_key = f"{self.env_name}_s{hash(str(self.state.observation)) % 10000}"
        
        self._current_state_key = state_key
        
        # Get features
        try:
            features = self.encoder.encode_features(self.state.observation)
        except:
            features = {}
        
        # ═══════════════════════════════════════════════════════════════════
        # PLAN-BASED: On episode start, get full map and create plan
        # ═══════════════════════════════════════════════════════════════════
        map_info = {}
        if self._use_planning and self._episode_just_started:
            if isinstance(self.game_handler, GridGameHandler):
                # Get full map report
                map_report = self.game_handler.get_map_report()
                
                # Feed to DriverNode with full map context
                if self.driver_node:
                    self.driver_node.see(map_report)
                
                # Create initial plan
                plan = self.game_handler.start_episode(self.state.observation)
                self._plan_initialized = True
                
                if plan:
                    logger.info(f"Episode {self.state.episode_count}: Plan created with {len(plan.steps)} steps")
                    map_info["has_plan"] = True
                    map_info["plan_steps"] = len(plan.steps)
                else:
                    logger.warning(f"Episode {self.state.episode_count}: No path found!")
                    map_info["has_plan"] = False
                
                # Include map info in perception
                map_info["grid_rows"] = self.game_handler.n_rows
                map_info["grid_cols"] = self.game_handler.n_cols
                map_info["holes"] = len(self.game_handler.holes)
                if self.game_handler.goal_pos:
                    map_info["goal"] = self.game_handler.goal_pos
            
            self._episode_just_started = False
        
        # Feed to DriverNode (regular perception)
        if self.driver_node and not map_info:  # Don't double-feed on episode start
            sensor = SensorReport.vision(
                description=state_key,
                objects=[{"type": state_key, "state": state_key}]
            )
            sensor.status = {"step": self.state.step_count, "episode": self.state.episode_count}
            self.driver_node.see(sensor)
        
        return Perception(
            entities=[state_key],
            locations=[self.env_name],
            properties={
                "state_key": state_key,
                "step": self.state.step_count,
                "episode": self.state.episode_count,
                "episode_reward": self.state.episode_reward,
                "last_reward": self.state.reward,
                **features,
                **map_info
            },
            events=[],
            heat_value=0.0,
            source_driver=self.DRIVER_ID,
            raw=self.state.observation
        )
    
    def act(self, action: Action) -> ActionResult:
        """Execute action in gym environment."""
        env = self.gym_env
        if not env or not self.decoder:
            return ActionResult(success=False, outcome="Not initialized", heat_value=0.0)
        
        action_name = action.target or action.action_type
        
        # Decode action
        try:
            if action_name in self._action_map:
                gym_action = self._action_map[action_name]
            else:
                gym_action = self.decoder.encode(action_name)
        except:
            gym_action = 0
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(gym_action)
        
        # Update state
        self.state.observation = obs
        self.state.reward = float(reward)
        self.state.terminated = terminated
        self.state.truncated = truncated
        self.state.info = info
        self.state.step_count += 1
        self.state.episode_reward += float(reward)
        
        done = terminated or truncated
        
        # Decay motor heat after action (prevents accumulation)
        self._decay_motor_heat()
        
        # Record outcome via game handler
        if self.game_handler:
            self.game_handler.record_outcome(self._current_state_key, action_name, float(reward), done)
        
        # ═══════════════════════════════════════════════════════════════════
        # MOTOR & RESULT LEARNING: Update nodes based on outcome
        # This feeds into psychology's decision making
        # ═══════════════════════════════════════════════════════════════════
        if done and self.manifold:
            # Update motor node heat
            motor_concept = f"{action_name}_motor"
            motor_node = self.manifold.get_node_by_concept(motor_concept)
            if motor_node:
                if reward > 0:
                    motor_node.add_heat(K * 0.5)  # Reward success
                else:
                    motor_node.heat = max(K * 0.1, motor_node.heat - K * 0.2)  # Punish failure
            
            # Create/update result node (e.g., "down_result")
            result_concept = f"{action_name}_result"
            result_node = self.manifold.get_node_by_concept(result_concept)
            if not result_node:
                result_node = Node(
                    concept=result_concept,
                    position="un",
                    heat=K * 0.5,
                    polarity=1 if reward > 0 else -1,
                    existence="actual",
                    righteousness=0.5
                )
                result_node.add_tag("result")
                result_node.add_tag(f"action:{action_name}")
                self.manifold.add_node(result_node)
            
            # Update result node based on outcome
            if reward > 0:
                result_node.add_heat(K * 0.3)
                result_node.polarity = 1
            else:
                result_node.heat = max(K * 0.1, result_node.heat - K * 0.1)
                result_node.polarity = -1
            
            logger.debug(f"Updated {result_concept}: heat={result_node.heat:.2f}, polarity={result_node.polarity}")
        
        # ═══════════════════════════════════════════════════════════════════
        # PLAN-BASED: Record plan outcome when episode ends
        # ═══════════════════════════════════════════════════════════════════
        if done and self._use_planning and isinstance(self.game_handler, GridGameHandler):
            plan_success = reward > -100 if "CliffWalking" in self.env_name else reward > 0
            self.game_handler.end_episode(plan_success)
        
        # ═══════════════════════════════════════════════════════════════════
        # GAME-SPECIFIC SUCCESS DETECTION
        # ═══════════════════════════════════════════════════════════════════
        # 
        # Different games have different reward semantics:
        #   CliffWalking: -1 = valid step (NEUTRAL), -100 = cliff (FAIL), done = goal (SUCCESS)
        #   FrozenLake:   0 = hole or step (NEUTRAL on step, FAIL on done), +1 = goal (SUCCESS)
        #   Blackjack:    -1 = loss, 0 = push, +1 = win
        #
        # We use success=None for neutral outcomes (no learning signal)
        
        if "CliffWalking" in self.env_name:
            # CliffWalking: -1 is normal step, -100 is cliff, goal is done without -100
            if reward == -100:
                success = False  # Fell off cliff
                heat_value = -1.0 * self.HEAT_SCALE
                success_type = "failure"
            elif done:
                success = True   # Reached goal!
                heat_value = 1.0 * self.HEAT_SCALE
                success_type = "success"
            else:
                success = None   # Neutral step (-1 reward)
                heat_value = 0.0
                success_type = "neutral"
        
        elif "FrozenLake" in self.env_name:
            # FrozenLake: +1 on goal, 0 on hole or step
            if done:
                success = reward > 0
                heat_value = float(reward) * self.HEAT_SCALE if reward > 0 else -0.5 * self.HEAT_SCALE
                success_type = "success" if success else "failure"
            else:
                success = None  # Just a step
                heat_value = 0.0
                success_type = "neutral"
        
        elif "Blackjack" in self.env_name:
            # Blackjack: +1 win, -1 loss, 0 push (only matters when done)
            if done:
                if reward > 0:
                    success = True
                    success_type = "success"
                elif reward < 0:
                    success = False
                    success_type = "failure"
                else:
                    success = None  # Push
                    success_type = "neutral"
                heat_value = float(reward) * self.HEAT_SCALE
            else:
                success = None
                heat_value = 0.0
                success_type = "neutral"
        
        else:
            # Generic: positive reward = success, negative = failure, zero = neutral
            if reward > 0:
                success = True
                success_type = "success"
            elif reward < 0:
                success = False
                success_type = "failure"
            else:
                success = None
                success_type = "neutral"
            heat_value = float(reward) * self.HEAT_SCALE
        
        result = ActionResult(
            success=success if success is not None else False,  # Default for routing
            outcome=f"{action_name} → reward={reward:.2f}" + (" [DONE]" if done else ""),
            heat_value=heat_value,
            changes={
                "reward": float(reward),
                "done": done,
                "terminated": terminated,
                "truncated": truncated,
                "step": self.state.step_count,
                "episode_reward": float(self.state.episode_reward),
                "success_type": success_type
            }
        )
        
        return result
    
    def reset(self):
        """Reset environment for new episode."""
        env = self.gym_env
        if env:
            if self.state.step_count > 0:
                logger.info(f"Episode ended: reward={self.state.episode_reward:.1f}")
            
            obs, info = env.reset()
            self.state = GymState(
                observation=obs,
                info=info,
                episode_count=self.state.episode_count + 1
            )
            
            # Mark episode as just started for plan initialization
            self._episode_just_started = True
            self._plan_initialized = False
    
    def get_available_actions(self) -> List[str]:
        """Get available actions."""
        if self.decoder:
            return self.decoder.action_names.copy()
        return []
    
    def supports_action(self, action_type: str) -> bool:
        """Check if action is supported."""
        return action_type in self._action_map or action_type in ["move", "interact", "observe", "wait"]
    
    def get_info(self) -> dict:
        """Get driver info."""
        return {
            "driver_id": self.DRIVER_ID,
            "driver_name": self.DRIVER_NAME,
            "env_name": self.env_name,
            "step_count": self.state.step_count,
            "episode_count": self.state.episode_count,
            "episode_reward": self.state.episode_reward,
            "actions": self.get_available_actions(),
            "handler": type(self.game_handler).__name__ if self.game_handler else "None"
        }
    
    def get_action_scores(self) -> Dict[str, float]:
        """
        Get scores for all actions, integrating with PBAI psychology.
        
        Flow:
        1. If plan has a clear next action → heat that motor → return scores
        2. If no plan or plan complete → return all 0.5 → falls through to DecisionNode
        3. DecisionNode uses psychology (collapse/correlate/select) and history
        
        This ensures:
        - Planned actions get preference via motor heat
        - Equal situations use psychology/history
        - Random exploration still happens via exploration_rate
        """
        if not self.game_handler:
            return {}
        
        available = self.get_available_actions()
        
        # ═══════════════════════════════════════════════════════════════════
        # PLAN-BASED: Check if plan has a clear next action
        # ═══════════════════════════════════════════════════════════════════
        planned_action = None
        if self._use_planning and isinstance(self.game_handler, GridGameHandler):
            planned_action = self.game_handler.get_next_action(self.state.observation)
        
        if planned_action and planned_action in available:
            # We have a plan with a clear action → heat it and return biased scores
            self.heat_planned_action(planned_action)
            logger.debug(f"Plan step {self.game_handler.plan_step}: {planned_action}")
            
            # Return scores with planned action clearly preferred
            scores = {}
            for action in available:
                if action == planned_action:
                    scores[action] = 0.85  # High but not absolute - allows some variation
                else:
                    scores[action] = 0.35  # Low but not zero - plan can be overridden
            return scores
        
        # ═══════════════════════════════════════════════════════════════════
        # NO PLAN: Return neutral scores → falls through to DecisionNode
        # DecisionNode uses psychology (Ego, history, collapse/correlate/select)
        # ═══════════════════════════════════════════════════════════════════
        logger.debug("No planned action - deferring to psychology/history")
        
        # Return all 0.5 so decide() falls through to DecisionNode.decide()
        # which uses psychology nodes and history
        return {action: 0.5 for action in available}
    
    def get_planned_action(self) -> Optional[str]:
        """Get the next action from the current plan (for grid games)."""
        if not self._use_planning or not isinstance(self.game_handler, GridGameHandler):
            return None
        
        return self.game_handler.get_next_action(self.state.observation)
    
    def has_active_plan(self) -> bool:
        """Check if there's an active plan being executed."""
        if not self._use_planning or not isinstance(self.game_handler, GridGameHandler):
            return False
        
        handler = self.game_handler
        return (handler.current_plan is not None and 
                handler.plan_step < len(handler.current_plan.steps))


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY & ALIASES
# ═══════════════════════════════════════════════════════════════════════════════

def create_gym_driver(env_name: str, manifold=None, **kwargs) -> GymDriver:
    """Factory function to create a GymDriver."""
    return GymDriver(env_name, manifold=manifold, **kwargs)


# Backward compatibility
GymDriverV2 = GymDriver
create_gym_driver_v2 = create_gym_driver
