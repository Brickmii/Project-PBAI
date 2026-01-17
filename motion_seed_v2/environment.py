"""
Motion Calendar Seed - Environment Module
Wraps external games/applications for agent interaction.

NO CHEATING:
- Cannot access internal game state directly
- Observes only visible GUI elements (text, button states)
- Discovers actions by inspection
- Learns effects through trial
"""

import time
import tkinter as tk
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import re


@dataclass
class Observation:
    """What the agent can see - only GUI-visible information."""
    visible_text: Dict[str, str]  # label_name -> text content
    button_states: Dict[str, bool]  # button_name -> enabled?
    numeric_values: Dict[str, float]  # extracted numbers
    timestamp: float = field(default_factory=time.time)
    raw_state: str = ""  # concatenated visible state for hashing
    
    def get_state_hash(self) -> str:
        """Create a hash of the visible state."""
        import hashlib
        return hashlib.sha256(self.raw_state.encode()).hexdigest()[:16]


@dataclass
class Action:
    """An action the agent can take."""
    name: str
    action_type: str  # "button", "key", "entry"
    target: Any  # The widget or key
    enabled: bool = True
    
    def __hash__(self):
        return hash((self.name, self.action_type))
    
    def __eq__(self, other):
        return self.name == other.name and self.action_type == other.action_type


@dataclass
class ActionResult:
    """Result of taking an action."""
    action: Action
    state_before: Observation
    state_after: Observation
    reward: float
    done: bool  # Episode ended?
    info: Dict[str, Any] = field(default_factory=dict)


class Environment(ABC):
    """Abstract base for game environments."""
    
    @abstractmethod
    def observe(self) -> Observation:
        """Get current visible state."""
        pass
    
    @abstractmethod
    def get_available_actions(self) -> List[Action]:
        """Get currently available actions."""
        pass
    
    @abstractmethod
    def execute_action(self, action: Action) -> ActionResult:
        """Execute an action and return result."""
        pass
    
    @abstractmethod
    def reset(self) -> Observation:
        """Reset environment to initial state."""
        pass
    
    @abstractmethod
    def compute_reward(self, before: Observation, after: Observation) -> float:
        """Compute reward from state transition."""
        pass


class TkinterGameEnvironment(Environment):
    """
    Wraps a tkinter game as an environment.
    
    Discovers widgets, observes text, executes actions.
    NO access to internal game logic.
    """
    
    def __init__(self, root: tk.Tk, game_instance: Any):
        self.root = root
        self.game = game_instance
        
        # Widget discovery
        self.buttons: Dict[str, tk.Button] = {}
        self.labels: Dict[str, tk.Label] = {}
        self.entries: Dict[str, tk.Entry] = {}
        self.canvas: Optional[tk.Canvas] = None
        
        # Action discovery
        self.discovered_actions: Dict[str, Action] = {}
        self.key_bindings: List[str] = []
        
        # State tracking (for reward computation)
        self.episode_start_state: Optional[Observation] = None
        self.last_observation: Optional[Observation] = None
        
        # Discover widgets
        self._discover_widgets(root)
        self._discover_key_bindings()
    
    def _discover_widgets(self, widget: tk.Widget, prefix: str = ""):
        """Recursively discover all widgets."""
        for child in widget.winfo_children():
            name = f"{prefix}_{child.winfo_class()}_{len(self.buttons) + len(self.labels)}"
            
            if isinstance(child, tk.Button):
                # Try to get button text
                try:
                    text = child.cget("text")
                    name = f"btn_{text.lower().replace(' ', '_')}"
                except:
                    pass
                self.buttons[name] = child
                
            elif isinstance(child, tk.Label):
                try:
                    text = child.cget("text")
                    if text:
                        name = f"lbl_{text[:20].lower().replace(' ', '_')}"
                except:
                    pass
                self.labels[name] = child
                
            elif isinstance(child, tk.Entry):
                name = f"entry_{len(self.entries)}"
                self.entries[name] = child
                
            elif isinstance(child, tk.Canvas):
                self.canvas = child
            
            # Recurse into children
            self._discover_widgets(child, name)
    
    def _discover_key_bindings(self):
        """Discover what keys are bound."""
        # Common game keys
        common_keys = ['Up', 'Down', 'Left', 'Right', 'w', 'a', 's', 'd', 
                       'W', 'A', 'S', 'D', 'space', 'Return', 'Escape']
        
        for key in common_keys:
            try:
                bindings = self.root.bind(f"<{key}>") or self.root.bind(f"<KeyPress-{key}>")
                if bindings:
                    self.key_bindings.append(key)
            except:
                pass
        
        # Also check for generic KeyPress binding
        if self.root.bind("<KeyPress>"):
            # Add common movement keys if generic binding exists
            self.key_bindings.extend(['Up', 'Down', 'Left', 'Right', 'w', 'a', 's', 'd'])
        
        self.key_bindings = list(set(self.key_bindings))
    
    def observe(self) -> Observation:
        """Observe the current visible state."""
        self.root.update()
        
        visible_text = {}
        numeric_values = {}
        
        # Read all label texts
        for name, label in self.labels.items():
            try:
                text = label.cget("text")
                visible_text[name] = str(text)
                
                # Extract numbers
                numbers = re.findall(r'-?\d+\.?\d*', str(text))
                for i, num in enumerate(numbers):
                    try:
                        numeric_values[f"{name}_num{i}"] = float(num)
                    except:
                        pass
            except:
                pass
        
        # Get button states
        button_states = {}
        for name, button in self.buttons.items():
            try:
                state = button.cget("state")
                button_states[name] = (state != tk.DISABLED)
            except:
                button_states[name] = True
        
        # Create raw state string for hashing
        raw_parts = [f"{k}:{v}" for k, v in sorted(visible_text.items())]
        raw_parts.extend([f"{k}:{v}" for k, v in sorted(button_states.items())])
        raw_state = "|".join(raw_parts)
        
        obs = Observation(
            visible_text=visible_text,
            button_states=button_states,
            numeric_values=numeric_values,
            raw_state=raw_state
        )
        
        self.last_observation = obs
        return obs
    
    def get_available_actions(self) -> List[Action]:
        """Get currently available actions."""
        actions = []
        
        # Button actions
        for name, button in self.buttons.items():
            try:
                state = button.cget("state")
                enabled = (state != tk.DISABLED)
                action = Action(
                    name=name,
                    action_type="button",
                    target=button,
                    enabled=enabled
                )
                actions.append(action)
                self.discovered_actions[name] = action
            except:
                pass
        
        # Key actions
        for key in self.key_bindings:
            action = Action(
                name=f"key_{key.lower()}",
                action_type="key",
                target=key,
                enabled=True
            )
            actions.append(action)
            self.discovered_actions[f"key_{key.lower()}"] = action
        
        # Entry actions (typing)
        for name, entry in self.entries.items():
            action = Action(
                name=f"type_{name}",
                action_type="entry",
                target=entry,
                enabled=True
            )
            actions.append(action)
        
        return actions
    
    def execute_action(self, action: Action, entry_value: str = None) -> ActionResult:
        """Execute an action and observe result."""
        state_before = self.observe()
        
        try:
            if action.action_type == "button":
                # Click button
                button = action.target
                if button.cget("state") != tk.DISABLED:
                    button.invoke()
            
            elif action.action_type == "key":
                # Simulate key press
                key = action.target
                event = tk.Event()
                event.keysym = key
                event.char = key if len(key) == 1 else ''
                event.keycode = 0
                self.root.event_generate(f"<KeyPress-{key}>")
            
            elif action.action_type == "entry" and entry_value is not None:
                # Type into entry
                entry = action.target
                entry.delete(0, tk.END)
                entry.insert(0, entry_value)
            
            # Let GUI update
            self.root.update()
            time.sleep(0.05)  # Small delay for animations
            self.root.update()
            
        except Exception as e:
            pass
        
        state_after = self.observe()
        reward = self.compute_reward(state_before, state_after)
        
        # Check if episode done
        done = self._check_episode_done(state_after)
        
        return ActionResult(
            action=action,
            state_before=state_before,
            state_after=state_after,
            reward=reward,
            done=done
        )
    
    def reset(self) -> Observation:
        """Reset the environment."""
        # Look for reset/new game buttons
        for name, action in self.discovered_actions.items():
            if any(word in name.lower() for word in ['new', 'reset', 'restart', 'deal']):
                if action.action_type == "button":
                    try:
                        action.target.invoke()
                        self.root.update()
                    except:
                        pass
                    break
        
        obs = self.observe()
        self.episode_start_state = obs
        return obs
    
    def compute_reward(self, before: Observation, after: Observation) -> float:
        """
        Compute reward from state transition.
        
        This is game-agnostic - looks for:
        - Numbers going up (often good)
        - Win/victory text appearing
        - Lose/bust text appearing
        """
        reward = 0.0
        
        # Check for numeric changes
        for key in after.numeric_values:
            if key in before.numeric_values:
                delta = after.numeric_values[key] - before.numeric_values[key]
                if 'balance' in key.lower() or 'score' in key.lower():
                    reward += delta * 0.01  # Scale down
        
        # Check for win/lose keywords in new text
        after_text = " ".join(after.visible_text.values()).lower()
        before_text = " ".join(before.visible_text.values()).lower()
        
        new_text = after_text.replace(before_text, "")
        
        if any(word in new_text for word in ['win', 'won', 'victory', 'success', 'goal']):
            reward += 1.0
        if any(word in new_text for word in ['lose', 'lost', 'bust', 'busted', 'game over']):
            reward -= 0.5
        
        return reward
    
    def _check_episode_done(self, obs: Observation) -> bool:
        """Check if the episode has ended."""
        text = " ".join(obs.visible_text.values()).lower()
        
        # Check for end-of-episode indicators
        if any(word in text for word in ['victory', 'game over', 'you win', 'you lose']):
            return True
        
        return False


class MazeEnvironment(TkinterGameEnvironment):
    """Specialized environment for maze games."""
    
    def __init__(self, root: tk.Tk, game_instance: Any):
        super().__init__(root, game_instance)
        self.prev_position = None
        self.visited_positions = set()
        self.steps_taken = 0
    
    def compute_reward(self, before: Observation, after: Observation) -> float:
        """Maze-specific reward."""
        reward = super().compute_reward(before, after)
        
        # Small negative reward for each step (encourages efficiency)
        reward -= 0.01
        self.steps_taken += 1
        
        # Check for goal reached
        after_text = " ".join(after.visible_text.values()).lower()
        if 'goal' in after_text and 'victory' in after_text:
            reward += 10.0
        
        return reward
    
    def reset(self) -> Observation:
        """Reset maze."""
        self.visited_positions = set()
        self.steps_taken = 0
        return super().reset()


class BlackjackEnvironment(TkinterGameEnvironment):
    """Specialized environment for blackjack."""
    
    def __init__(self, root: tk.Tk, game_instance: Any):
        super().__init__(root, game_instance)
        self.hands_played = 0
        self.starting_balance = None
    
    def compute_reward(self, before: Observation, after: Observation) -> float:
        """Blackjack-specific reward."""
        reward = 0.0
        
        # Track balance changes
        balance_before = None
        balance_after = None
        
        for key, val in before.numeric_values.items():
            if 'balance' in key.lower():
                balance_before = val
        
        for key, val in after.numeric_values.items():
            if 'balance' in key.lower():
                balance_after = val
        
        if balance_before is not None and balance_after is not None:
            delta = balance_after - balance_before
            reward = delta / 100.0  # Scale
        
        # Check for win/lose in text
        after_text = " ".join(after.visible_text.values()).lower()
        
        if 'blackjack!' in after_text:
            reward += 0.5
        if 'bust' in after_text:
            reward -= 0.3
        
        return reward
    
    def reset(self) -> Observation:
        """Reset for new hand."""
        # Find and click deal button
        for name, action in self.discovered_actions.items():
            if 'deal' in name.lower():
                if action.action_type == "button":
                    # First set a bet
                    for entry_name, entry in self.entries.items():
                        entry.delete(0, tk.END)
                        entry.insert(0, "10")
                    
                    try:
                        action.target.invoke()
                        self.root.update()
                    except:
                        pass
                    break
        
        self.hands_played += 1
        return self.observe()


def create_environment(game_type: str, root: tk.Tk, game_instance: Any) -> Environment:
    """Factory function to create appropriate environment."""
    if game_type.lower() == "maze":
        return MazeEnvironment(root, game_instance)
    elif game_type.lower() == "blackjack":
        return BlackjackEnvironment(root, game_instance)
    else:
        return TkinterGameEnvironment(root, game_instance)
