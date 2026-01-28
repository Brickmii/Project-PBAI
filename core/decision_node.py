"""
PBAI Decision Node - Movement (Lin) - The 6th Motion Function

════════════════════════════════════════════════════════════════════════════════
THE 6 MOTION FUNCTIONS → THE 6 CORE FILES
════════════════════════════════════════════════════════════════════════════════

    1. Heat (Σ)           - psychology     - magnitude validator
    2. Polarity (+/-)     - node_constants - direction validator  
    3. Existence (δ)      - clock_node     - persistence validator (Self IS clock)
    4. Righteousness (R)  - nodes          - alignment validator (frames)
    5. Order (Q)          - manifold       - arithmetic validator
                            ↓
    6. Movement (Lin)     - decision_node  - VECTORIZED OUTPUT (this file)

The decision node is the EXIT point - where 5 scalar inputs become 1 vector output.

════════════════════════════════════════════════════════════════════════════════
5 SCALARS → 1 VECTOR
════════════════════════════════════════════════════════════════════════════════

Decision takes the 5 validated scalars and produces movement:

    Heat (Σ)              How much energy for this action?
    Polarity (+/-)        Which direction? (approach/avoid)
    Existence (δ)         Does this option exist/persist? (above 1/φ³?)
    Righteousness (R)     Is this option aligned? (R→0?)
    Order (Q)             What's the history? (success count)
                          ↓
    Movement (Lin)        THE DECISION (selected action vector)

════════════════════════════════════════════════════════════════════════════════
THE 5/6 CONFIDENCE THRESHOLD
════════════════════════════════════════════════════════════════════════════════

    Confidence = Conscience's mediation (Identity → Ego)
    
    When confidence > 5/6 (0.8333):
        - 5 scalar functions are validated
        - EXPLOIT: Use the learned pattern
        
    When confidence < 5/6:
        - Still gathering validation
        - EXPLORE: Try to learn more

    t = 5K validations crosses threshold (one K-quantum per scalar)

════════════════════════════════════════════════════════════════════════════════
DECISION PROCESS (Collapse → Correlate → Select)
════════════════════════════════════════════════════════════════════════════════

    1. COLLAPSE: Find CENTER node (R→0, most righteous)
    2. CORRELATE: Get CLUSTER from center (current + historical + novel)
    3. SELECT: 
       - If confidence > 5/6: EXPLOIT Order (use proven pattern)
       - If confidence < 5/6: EXPLORE (try options, learn)

════════════════════════════════════════════════════════════════════════════════
ARCHITECTURE
════════════════════════════════════════════════════════════════════════════════

    ┌─────────────────────────────────────────────────────────────┐
    │                      MANIFOLD                                │
    │                                                              │
    │   [environment.py]   ──────────▶  [decision_node.py]        │
    │      (ENTRY)            process       (EXIT)                 │
    │           │                              │                   │
    │           ▼                              ▼                   │
    │   ┌─────────────┐               ┌─────────────┐             │
    │   │  Identity   │ ────────────▶ │    Ego      │             │
    │   │  (observe)  │   Conscience  │  (decide)   │             │
    │   └─────────────┘   (mediate)   └─────────────┘             │
    └──────────┬───────────────────────────────┬──────────────────┘
               │                               │
               │ Perception                    │ Action (Movement)
               ▼                               ▼
    ┌─────────────────┐               ┌─────────────────┐
    │   driver node    │               │   driver node    │
    │   (states)      │               │   (plans)       │
    └─────────────────┘               └─────────────────┘

════════════════════════════════════════════════════════════════════════════════
"""

import os
import json
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from time import time

from .nodes import Node, Axis, Order, Element
from .node_constants import (
    K, PHI, THRESHOLD_ORDER, THRESHOLD_EXISTENCE, 
    CONFIDENCE_EXPLOIT_THRESHOLD,
    EXISTENCE_ACTUAL, EXISTENCE_DORMANT, EXISTENCE_POTENTIAL,
    get_growth_path
)
from .driver_node import MotorAction, ActionPlan, SensorReport

logger = logging.getLogger(__name__)


def get_decisions_path() -> str:
    """Get path to decisions folder."""
    project_root = get_growth_path("").replace("/growth", "")
    return os.path.join(project_root, "decisions")


@dataclass
class Choice:
    """
    A recorded choice with its outcome - the atomic unit of decision.
    
    Records the 5 scalar inputs and 1 vector output:
    
        INPUTS (5 scalars):
        - heat: Energy available (magnitude)
        - polarity: Direction preference (+1 approach, -1 avoid)
        - existence: Does option persist? (above 1/φ³?)
        - righteousness: Is option aligned? (R value)
        - order: Historical success count
        
        OUTPUT (1 vector):
        - selected: The chosen action (movement)
    
    Context enables generalization - features like "near_cliff" can be
    shared across multiple state_keys, allowing learning to transfer.
    """
    timestamp: float
    state_key: str                      # What state we were in
    options: List[str]                  # What options were available
    selected: str                       # OUTPUT: The movement vector
    confidence: float                   # Ego's confidence (via Conscience)
    
    # The 5 scalar inputs (optional, for detailed tracking)
    heat: float = 0.0                   # 1. Heat (Σ) - magnitude
    polarity: int = 1                   # 2. Polarity (+/-) - direction
    existence_valid: bool = True        # 3. Existence (δ) - above 1/φ³?
    righteousness: float = 1.0          # 4. Righteousness (R) - alignment
    order_count: int = 0                # 5. Order (Q) - success history
    
    # Context and outcome
    context: Dict[str, Any] = field(default_factory=dict)
    outcome: Optional[str] = None
    success: Optional[bool] = None
    heat_delta: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "state_key": self.state_key,
            "options": self.options,
            "selected": self.selected,
            "confidence": self.confidence,
            "heat": self.heat,
            "polarity": self.polarity,
            "existence_valid": self.existence_valid,
            "righteousness": self.righteousness,
            "order_count": self.order_count,
            "context": self.context,
            "outcome": self.outcome,
            "success": self.success,
            "heat_delta": self.heat_delta
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> 'Choice':
        return cls(
            timestamp=data["timestamp"],
            state_key=data["state_key"],
            options=data["options"],
            selected=data["selected"],
            confidence=data["confidence"],
            heat=data.get("heat", 0.0),
            polarity=data.get("polarity", 1),
            existence_valid=data.get("existence_valid", True),
            righteousness=data.get("righteousness", 1.0),
            order_count=data.get("order_count", 0),
            context=data.get("context", {}),
            outcome=data.get("outcome"),
            success=data.get("success"),
            heat_delta=data.get("heat_delta", 0.0)
        )


class ChoiceNode:
    """
    A node that records choices over time.
    
    Base class for DecisionNode (exit) and EnvironmentNode (entry).
    Saves choice history to filesystem.
    
    Structure:
        Node (choice_node)
        ├── Axis: "history" with Order
        │   └── Elements: previous choices (with heat = success)
        ├── Axis: "patterns" 
        │   └── State → Choice mappings with traversal count
        └── Connection to driver node
    """
    
    def __init__(self, name: str, manifold: 'Manifold', driver: Optional[Node] = None,
                 save_dir: str = None):
        self.name = name
        self.manifold = manifold
        self.driver = driver
        self.born = False  # Birth tracking
        self.save_dir = save_dir or get_decisions_path()
        
        # Core node (will be set during birth)
        self.node = None
        
        # In-memory choice buffer
        self.choices: List[Choice] = []
        self.max_history = 1000
        
        # Birth
        self._birth()
    
    def _birth(self):
        """Birth this choice node - create node and load history."""
        if self.born:
            logger.warning(f"ChoiceNode {self.name} already born, skipping")
            return
        
        # Create or find node
        existing = self.manifold.get_node_by_concept(self.name) if self.manifold else None
        if existing:
            self.node = existing
        else:
            self.node = Node(
                concept=self.name,
                position="u",
                heat=K,
                polarity=1,
                existence="actual",
                righteousness=1.0,  # Righteous frame
                order=1
            )
            if self.manifold:
                self.manifold.add_node(self.node)
        
        # Ensure history axis exists
        if not self.node.get_axis("history"):
            history_axis = self.node.add_axis("history", f"{self.name}_history")
            history_axis.make_proper()  # Has Order
        
        # Load from filesystem
        self._load()
        
        self.born = True
        logger.debug(f"ChoiceNode {self.name} born")
    
    def _get_filepath(self) -> str:
        """Get path for this choice node's data."""
        os.makedirs(self.save_dir, exist_ok=True)
        safe_name = self.name.replace("/", "_").replace("\\", "_")
        return os.path.join(self.save_dir, f"{safe_name}.json")
    
    def _load(self):
        """Load choices from filesystem."""
        filepath = self._get_filepath()
        if os.path.exists(filepath):
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                self.choices = [Choice.from_dict(c) for c in data.get("choices", [])]
                logger.debug(f"Loaded {len(self.choices)} choices for {self.name}")
            except Exception as e:
                logger.warning(f"Failed to load choices for {self.name}: {e}")
    
    def save(self):
        """Save choices to filesystem."""
        filepath = self._get_filepath()
        data = {
            "name": self.name,
            "choices": [c.to_dict() for c in self.choices],
            "total_choices": len(self.choices),
            "success_count": sum(1 for c in self.choices if c.success),
            "saved_at": time()
        }
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        logger.debug(f"Saved {len(self.choices)} choices for {self.name}")
    
    def record(self, choice: Choice):
        """
        Record a choice.
        
        Creates axes for:
        1. state_key → tracks this specific state
        2. state_key_action → tracks this action in this state  
        3. context_item_action → tracks this action with this context (GENERALIZATION)
        """
        self.choices.append(choice)
        if len(self.choices) > self.max_history:
            self.choices.pop(0)
        
        # Add to node's history Order
        history_axis = self.node.get_axis("history")
        if history_axis and history_axis.order:
            element = Element(
                node_id=f"choice_{len(history_axis.order.elements)}",
                index=1 if choice.success else 0
            )
            history_axis.order.elements.append(element)
        
        # Update pattern strength (state → choice mapping)
        pattern_axis = self.node.get_axis(choice.state_key)
        if pattern_axis is None:
            pattern_axis = self.node.add_axis(choice.state_key, f"{self.name}_{choice.state_key}")
        
        # Track which choices work in this state
        choice_key = f"{choice.state_key}_{choice.selected}"
        choice_axis = self.node.get_axis(choice_key)
        if choice_axis is None:
            choice_axis = self.node.add_axis(choice_key, f"{self.name}_{choice_key}")
            choice_axis.make_proper()
        
        # Record outcome in state_action Order
        if choice_axis.order is None:
            choice_axis.make_proper()
        choice_axis.order.elements.append(
            Element(node_id=f"{choice_key}_{len(choice_axis.order.elements)}", 
                    index=1 if choice.success else 0)
        )
        
        # Adjust traversal_count based on outcome
        if choice.success is not None:
            if choice.success:
                choice_axis.traversal_count += 1
        
        # CONTEXT-BASED LEARNING (enables generalization)
        # Create axes for context_item + action combinations
        for ctx_key, ctx_value in choice.context.items():
            # Only track "active" context items (truthy values)
            if ctx_value:
                context_choice_key = f"ctx:{ctx_key}_{choice.selected}"
                ctx_axis = self.node.get_axis(context_choice_key)
                if ctx_axis is None:
                    ctx_axis = self.node.add_axis(context_choice_key, f"{self.name}_{context_choice_key}")
                    ctx_axis.make_proper()
                
                # Record outcome
                if ctx_axis.order is None:
                    ctx_axis.make_proper()
                ctx_axis.order.elements.append(
                    Element(node_id=f"{context_choice_key}_{len(ctx_axis.order.elements)}",
                            index=1 if choice.success else 0)
                )
                
                # Adjust traversal_count
                if choice.success is not None:
                    if choice.success:
                        ctx_axis.traversal_count += 1
                
                logger.debug(f"Context learning: {ctx_key}+{choice.selected} → {'✓' if choice.success else '✗'}")
        
        # Auto-save
        self.save()
    
    def get_best_choice(self, state_key: str, options: List[str], 
                        context: Dict[str, Any] = None) -> Optional[str]:
        """
        Get the historically best choice for a state.
        
        Considers:
        1. State-specific history (state_key + action)
        2. Context-based history (context_item + action) - enables generalization
        
        Args:
            state_key: Current state
            options: Available options
            context: Current context (features like near_cliff)
            
        Returns:
            Best option based on combined history, or None if no history
        """
        if context is None:
            context = {}
        
        # Score each option
        option_scores = {}
        
        for option in options:
            score = 0.0
            weight = 0.0
            
            # 1. State-specific score (weighted 0.6)
            state_axis = self.node.get_axis(f"{state_key}_{option}")
            if state_axis:
                state_score = self._score_from_axis(state_axis)
                if state_score is not None:
                    score += state_score * 0.6
                    weight += 0.6
            
            # 2. Context-based scores (weighted 0.4 total, split among active contexts)
            active_contexts = [k for k, v in context.items() if v]
            if active_contexts:
                ctx_weight_each = 0.4 / len(active_contexts)
                for ctx_key in active_contexts:
                    ctx_axis = self.node.get_axis(f"ctx:{ctx_key}_{option}")
                    if ctx_axis:
                        ctx_score = self._score_from_axis(ctx_axis)
                        if ctx_score is not None:
                            score += ctx_score * ctx_weight_each
                            weight += ctx_weight_each
            
            if weight > 0:
                option_scores[option] = score / weight
        
        if not option_scores:
            return None
        
        # Return best scoring option
        return max(option_scores, key=option_scores.get)
    
    def _score_from_axis(self, axis: Axis) -> Optional[float]:
        """
        Score an action from its axis history.
        
        Uses:
        - Traversal count (experience)
        - Success rate from Order elements
        
        Returns:
            Score 0.0-1.0, or None if no data
        """
        if not axis.order or not axis.order.elements:
            # Use traversal count as proxy if no Order
            if axis.traversal_count > 0:
                return min(axis.traversal_count / 10.0, 1.0)
            return None
        
        # Calculate success rate
        successes = sum(1 for e in axis.order.elements if e.index == 1)
        total = len(axis.order.elements)
        success_rate = successes / total if total > 0 else 0.5
        
        # Weight by confidence (more samples = more confident)
        confidence = min(total / 10.0, 1.0)
        
        # Blend success rate with prior (0.5) based on confidence
        return success_rate * confidence + 0.5 * (1 - confidence)
    
    def get_success_rate(self, state_key: str = None, choice: str = None) -> float:
        """Get success rate for a state or specific choice."""
        relevant = self.choices
        
        if state_key:
            relevant = [c for c in relevant if c.state_key == state_key]
        if choice:
            relevant = [c for c in relevant if c.selected == choice]
        
        if not relevant:
            return 0.0
        
        successes = sum(1 for c in relevant if c.success)
        return successes / len(relevant)


class DecisionNode(ChoiceNode):
    """
    Movement (Lin) - The 6th Motion Function - Vectorized Output.
    
    This is the EXIT point where 5 scalar inputs become 1 vector output:
    
        INPUTS (5 scalars, validated by other files):
        ─────────────────────────────────────────────
        1. Heat (Σ)           ← psychology (magnitude)
        2. Polarity (+/-)     ← node_constants (direction)
        3. Existence (δ)      ← clock_node (persistence, 1/φ³ threshold)
        4. Righteousness (R)  ← nodes (alignment, R→0)
        5. Order (Q)          ← manifold (history, success count)
        
        OUTPUT (1 vector):
        ──────────────────
        6. Movement (Lin)     → THE DECISION (selected action)
    
    THE 5/6 CONFIDENCE THRESHOLD:
        - confidence > 5/6: EXPLOIT (5 scalars validated → use pattern)
        - confidence < 5/6: EXPLORE (still gathering validation)
        - t = 5K validations crosses threshold
    
    Saves decision history to decisions/ folder.
    """
    
    def __init__(self, manifold: 'Manifold', driver: Optional[Node] = None):
        super().__init__("pbai_decisions", manifold, driver, get_decisions_path())
        
        # Current decision context
        self.pending_choice: Optional[Choice] = None
    
    def begin_decision(self, state_key: str, options: List[str], confidence: float,
                       context: Dict[str, Any] = None) -> Choice:
        """
        Begin a decision - gather the 5 scalar inputs.
        
        Collects validation from all 5 scalar motion functions:
        1. Heat (Σ) - from psychology nodes
        2. Polarity (+/-) - from state context
        3. Existence (δ) - is state above 1/φ³?
        4. Righteousness (R) - state alignment
        5. Order (Q) - historical success count
        
        Args:
            state_key: Current state from perception
            options: Available choices
            confidence: Ego's confidence (via Conscience mediation)
            context: Features for generalization
        """
        # Gather the 5 scalar inputs
        heat = 0.0
        polarity = 1
        existence_valid = True
        righteousness = 1.0
        order_count = 0
        
        if self.manifold:
            # 1. Heat (Σ) - from Ego (decision-maker)
            if self.manifold.ego_node:
                heat = self.manifold.ego_node.heat
            
            # 2-5. From state node if it exists
            state_node = self.manifold.get_node_by_concept(state_key)
            if state_node:
                # 2. Polarity (+/-) - direction
                polarity = state_node.polarity
                
                # 3. Existence (δ) - above 1/φ³?
                existence_valid = state_node.existence == EXISTENCE_ACTUAL
                
                # 4. Righteousness (R) - alignment
                righteousness = state_node.righteousness
                
                # 5. Order (Q) - count choices for this state
                choice_axis = self.node.get_axis(state_key) if self.node else None
                if choice_axis:
                    order_count = choice_axis.traversal_count
        
        self.pending_choice = Choice(
            timestamp=time(),
            state_key=state_key,
            options=options,
            selected="",
            confidence=confidence,
            heat=heat,
            polarity=polarity,
            existence_valid=existence_valid,
            righteousness=righteousness,
            order_count=order_count,
            context=context or {}
        )
        return self.pending_choice
    
    def commit_decision(self, selected: str) -> Choice:
        """
        Commit to a decision.
        
        Args:
            selected: The chosen option
        """
        if self.pending_choice is None:
            # Create choice retroactively
            self.pending_choice = Choice(
                timestamp=time(),
                state_key="unknown",
                options=[selected],
                selected=selected,
                confidence=K
            )
        else:
            self.pending_choice.selected = selected
        
        logger.debug(f"Decision: {selected} (confidence={self.pending_choice.confidence:.2f})")
        return self.pending_choice
    
    def complete_decision(self, outcome: str, success: bool, heat_delta: float = 0.0):
        """
        Complete a decision with its outcome.
        
        This is called after the action has been executed and we know the result.
        """
        if self.pending_choice is None:
            logger.warning("No pending choice to complete")
            return
        
        self.pending_choice.outcome = outcome
        self.pending_choice.success = success
        self.pending_choice.heat_delta = heat_delta
        
        # Record in history (auto-saves)
        self.record(self.pending_choice)
        
        # Update driver if connected
        if self.driver and success:
            self.driver.reward(heat_delta, achieved=outcome)
        elif self.driver and not success:
            self.driver.punish(abs(heat_delta))
        
        logger.info(f"Decision complete: {self.pending_choice.selected} → {outcome} ({'✓' if success else '✗'})")
        self.pending_choice = None
    
    def decide(self, state_key: str, options: List[str], 
               confidence: float = None, goal: str = None,
               context: Dict[str, Any] = None) -> str:
        """
        Make a decision - the 6th motion function (vectorized movement).
        
        Takes 5 scalar inputs and produces 1 vector output:
        
            Heat (Σ)              → Energy available (from psychology)
            Polarity (+/-)        → Direction preference (from node_constants)
            Existence (δ)         → Option persistence (above 1/φ³?)
            Righteousness (R)     → Option alignment (R→0?)
            Order (Q)             → Historical success (from manifold)
                                  ↓
            Movement (Lin)        → THE DECISION (this output)
        
        The 5/6 Confidence Threshold:
            confidence > 5/6 (0.8333) → EXPLOIT: Use validated pattern
            confidence < 5/6         → EXPLORE: Try options, learn more
        
        Args:
            state_key: Current state
            options: Available choices
            confidence: Ego's confidence (via Conscience mediation)
            goal: Optional goal to achieve
            context: Features for generalization
            
        Returns:
            Selected option (the movement vector)
        """
        from .node_constants import collapse_wave_function, correlate_cluster, select_from_cluster
        
        # Get confidence from manifold if not provided
        if confidence is None:
            confidence = self.manifold.get_confidence(state_key) if self.manifold else 0.0
        if context is None:
            context = {}
        
        # Begin decision with context
        self.begin_decision(state_key, options, confidence, context)
        
        # Determine explore/exploit mode based on 5/6 threshold
        should_exploit = confidence > CONFIDENCE_EXPLOIT_THRESHOLD
        
        logger.debug(f"Decision mode: {'EXPLOIT' if should_exploit else 'EXPLORE'} "
                    f"(confidence={confidence:.3f}, threshold={CONFIDENCE_EXPLOIT_THRESHOLD:.3f})")
        
        # 1. If EXPLOIT mode and we have history, use proven pattern
        if should_exploit:
            best = self.get_best_choice(state_key, options, context)
            if best:
                logger.info(f"EXPLOIT decision: {best} (confidence={confidence:.3f})")
                return self.commit_decision(best).selected
        
        # 2. Check driver for goal-directed plan
        if self.driver and goal:
            decision = self.driver.think(goal)
            if isinstance(decision, ActionPlan):
                self.driver.start_plan(decision)
                decision = self.driver.think()
            if isinstance(decision, MotorAction):
                for opt in options:
                    if opt in str(decision):
                        return self.commit_decision(opt).selected
        
        # 3. COLLAPSE: Find center via wave function (uses R)
        selected = self._decide_by_collapse(state_key, options, context)
        if selected:
            return self.commit_decision(selected).selected
        
        # 4. EXPLORE fallback: Check history even in explore mode
        best = self.get_best_choice(state_key, options, context)
        if best:
            return self.commit_decision(best).selected
        
        # 5. Check driver patterns
        if self.driver:
            driver_decision = self.driver.think()
            if driver_decision:
                for opt in options:
                    if opt in str(driver_decision):
                        return self.commit_decision(opt).selected
        
        # 6. Default: Random exploration (first option)
        import random
        selected = random.choice(options) if len(options) > 1 else options[0]
        logger.info(f"EXPLORE decision: {selected} (random, confidence={confidence:.3f})")
        return self.commit_decision(selected).selected
    
    def _decide_by_collapse(self, state_key: str, options: List[str],
                           context: Dict[str, Any] = None) -> Optional[str]:
        """
        Decide using wave function collapse and cluster correlation.
        
        1. Find candidate nodes (related to state_key)
        2. Collapse to find CENTER (R→0)
        3. Correlate cluster from center (current + historical + novel)
        4. Select option: exploit Order if exists, else explore
        
        Returns:
            Selected option, or None if no cluster context
        """
        from .node_constants import collapse_wave_function, correlate_cluster, select_from_cluster
        
        if not self.manifold:
            return None
        
        # Find candidate nodes related to current state
        candidates = self._find_candidate_nodes(state_key, context)
        
        if not candidates:
            return None
        
        # COLLAPSE: Find the center (node with R closest to 0)
        center_idx = collapse_wave_function(candidates, self.manifold)
        if center_idx < 0:
            return None
        
        center_node = candidates[center_idx]
        logger.debug(f"Collapse found center: {center_node.concept} (R={center_node.righteousness:.3f})")
        
        # CORRELATE: Get cluster (current + historical + novel)
        cluster = correlate_cluster(center_node, self.manifold, max_depth=3)
        
        if not cluster.get('all'):
            return None
        
        logger.debug(f"Cluster: {len(cluster['current'])} current, "
                    f"{len(cluster['historical'])} historical, "
                    f"{len(cluster['novel'])} novel")
        
        # SELECT: Exploit Order if exists, else explore
        selected_idx, reason = select_from_cluster(options, cluster, self.manifold)
        
        if selected_idx >= 0 and selected_idx < len(options):
            selected = options[selected_idx]
            logger.info(f"Collapse decision: {selected} ({reason}, cluster={len(cluster['all'])})")
            return selected
        
        return None
    
    def _find_candidate_nodes(self, state_key: str, 
                              context: Dict[str, Any] = None) -> List[Node]:
        """
        Find nodes related to current state for collapse.
        
        Candidates are nodes that might be the CENTER:
        - The current state node (if exists)
        - Similar state nodes (share prefix)
        - Context-related nodes
        """
        if not self.manifold:
            return []
        
        candidates = []
        
        # 1. Current state node
        state_node = self.manifold.get_node_by_concept(state_key)
        if state_node:
            candidates.append(state_node)
        
        # 2. Similar states (share prefix)
        # e.g., "blackjack_hard_16" might relate to "blackjack_hard_17"
        prefix = state_key.rsplit('_', 1)[0] if '_' in state_key else state_key
        for concept, node_id in self.manifold.nodes_by_concept.items():
            if concept != state_key and concept.startswith(prefix):
                node = self.manifold.get_node(node_id)
                if node and node.existence != "archived":
                    candidates.append(node)
        
        # 3. Context-related nodes
        if context:
            for ctx_key, ctx_value in context.items():
                if ctx_value:  # Only active contexts
                    ctx_node = self.manifold.get_node_by_concept(f"ctx:{ctx_key}")
                    if ctx_node:
                        candidates.append(ctx_node)
        
        # 4. Nodes connected to current decision node
        for axis in self.node.frame.axes.values():
            if axis.target_id:
                target = self.manifold.get_node(axis.target_id)
                if target and target not in candidates:
                    candidates.append(target)
        
        return candidates


class EnvironmentNode(ChoiceNode):
    """
    Records perceptions received through environment.py (the ENTRY point).
    
    Note: The actual ENTRY logic is in drivers/environment.py
    This class just records and manages perception history.
    
    Connects external perceptions to:
    1. Identity (what exists)
    2. driver node states (learned state patterns)
    3. Choice history (what's been seen before)
    """
    
    def __init__(self, manifold: 'Manifold', driver: Optional[Node] = None):
        # Perceptions also save to decisions/ folder (they're choices the environment made)
        super().__init__("pbai_perceptions", manifold, driver, get_decisions_path())
    
    def receive(self, state_key: str, description: str, novelty: float = 0.0) -> Choice:
        """
        Receive a perception from the environment.
        
        Records what was perceived as a "choice" (the environment chose to show us this).
        
        Args:
            state_key: Unique key for this state
            description: Human-readable description
            novelty: How novel this perception is (affects heat)
        """
        # Perception is like a choice the environment made
        choice = Choice(
            timestamp=time(),
            state_key=state_key,
            options=[state_key],  # Only one "option" - what we perceived
            selected=state_key,
            confidence=novelty,  # Novelty = confidence in importance
            success=True,  # Perception is always "successful"
            heat_delta=novelty * THRESHOLD_ORDER
        )
        
        self.record(choice)
        
        # Driver node handling - now just tags, no .see() method
        # The sensor report would be processed elsewhere
        
        # Update Identity in manifold
        if self.manifold and hasattr(self.manifold, 'update_identity'):
            is_novel = novelty > 0.5
            self.manifold.update_identity(state_key, heat_delta=novelty, known=not is_novel)
        
        logger.debug(f"Perception: {state_key} (novelty={novelty:.2f})")
        return choice
    
    def is_familiar(self, state_key: str) -> bool:
        """Check if we've seen this state before."""
        return any(c.state_key == state_key for c in self.choices)
    
    def get_familiarity(self, state_key: str) -> float:
        """Get how familiar a state is (0-1)."""
        count = sum(1 for c in self.choices if c.state_key == state_key)
        # Diminishing returns on familiarity
        return min(1.0, count / 10.0)


# ═══════════════════════════════════════════════════════════════════════════════
# INTEGRATION - Connecting Entry and Exit
# ═══════════════════════════════════════════════════════════════════════════════

class PBAILoop:
    """
    The complete perception → decision → action loop.
    
    Entry: drivers/environment.py (perceptions come in)
    Exit: DecisionNode (actions go out)
    
    Connects:
    - EnvironmentNode (perception recording)
    - DecisionNode (exit/action)
    - driver node (learning/execution)
    - Manifold (core psychology)
    """
    
    def __init__(self, manifold: 'Manifold', driver: Optional[Node]):
        self.manifold = manifold
        self.driver = driver
        self.born = False  # Birth tracking
        
        # Entry and exit nodes (will be set during birth)
        self.entry = None
        self.exit = None
        
        # Birth
        self._birth()
    
    def _birth(self):
        """Birth this PBAI loop - create entry and exit nodes."""
        if self.born:
            logger.warning("PBAILoop already born, skipping")
            return
        
        # Create entry and exit nodes (both save to decisions/)
        self.entry = EnvironmentNode(self.manifold, self.driver)
        self.exit = DecisionNode(self.manifold, self.driver)
        
        self.born = True
        logger.info(f"PBAILoop born with driver: {self.driver.name}")
    
    def step(self, perception_key: str, perception_desc: str,
             options: List[str], goal: str = None) -> Tuple[str, Choice]:
        """
        One complete loop iteration.
        
        Args:
            perception_key: State key for current perception
            perception_desc: Human description of perception
            options: Available action options
            goal: Optional goal to work toward
            
        Returns:
            (selected_action, decision_choice)
        """
        # ENTRY: Receive perception
        novelty = 0.0 if self.entry.is_familiar(perception_key) else 1.0
        self.entry.receive(perception_key, perception_desc, novelty)
        
        # Get confidence from manifold
        confidence = self.manifold.get_confidence() if self.manifold else K
        
        # EXIT: Make decision
        selected = self.exit.decide(perception_key, options, confidence, goal)
        
        return selected, self.exit.pending_choice
    
    def complete(self, outcome: str, success: bool, heat_delta: float = 0.0):
        """Complete the current decision with outcome."""
        self.exit.complete_decision(outcome, success, heat_delta)
    
    def save(self):
        """Save state."""
        if self.driver:
            self.driver.save()
        if self.manifold:
            self.manifold.save_growth_map()
        # Choice nodes auto-save on record
