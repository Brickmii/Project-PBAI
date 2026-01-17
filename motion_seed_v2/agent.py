"""
Motion Calendar Seed - Agent Module
Learns to play games through exploration and the seed architecture.

NO CHEATING:
- Learns action effects through trial
- Builds strategy from experience
- Uses seed memory for state-action associations
"""

import random
import time
import math
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import hashlib

from environment import Environment, Action, Observation, ActionResult


@dataclass
class Experience:
    """A single experience tuple."""
    state_hash: str
    action_name: str
    next_state_hash: str
    reward: float
    done: bool
    timestamp: float = field(default_factory=time.time)


@dataclass
class ActionValue:
    """Q-value tracking for an action."""
    total_reward: float = 0.0
    count: int = 0
    last_reward: float = 0.0
    
    @property
    def average(self) -> float:
        return self.total_reward / self.count if self.count > 0 else 0.0
    
    def update(self, reward: float):
        self.count += 1
        self.total_reward += reward
        self.last_reward = reward


class GameAgent:
    """
    An agent that learns to play games.
    
    Uses exploration to discover:
    - What actions exist
    - What effects they have
    - What strategies work
    
    Integrates with seed for persistent learning.
    """
    
    def __init__(self, env: Environment, seed=None):
        self.env = env
        self.seed = seed  # Optional MotionSeed for persistence
        
        # Learning parameters
        self.epsilon = 1.0  # Exploration rate (starts high)
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.05
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        
        # Q-table: state_hash -> action_name -> ActionValue
        self.q_table: Dict[str, Dict[str, ActionValue]] = defaultdict(
            lambda: defaultdict(ActionValue)
        )
        
        # Experience replay
        self.experiences: List[Experience] = []
        self.max_experiences = 10000
        
        # Action discovery
        self.known_actions: Dict[str, Action] = {}
        self.action_effects: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )  # action -> outcome_description -> count
        
        # Statistics
        self.total_episodes = 0
        self.total_steps = 0
        self.total_reward = 0.0
        self.episode_rewards: List[float] = []
        
        # Current episode tracking
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
    
    def discover_actions(self) -> List[Action]:
        """Discover what actions are available."""
        actions = self.env.get_available_actions()
        
        for action in actions:
            self.known_actions[action.name] = action
        
        return actions
    
    def select_action(self, state: Observation) -> Optional[Action]:
        """
        Select action using epsilon-greedy strategy.
        
        - With probability epsilon: explore (random action)
        - Otherwise: exploit (best known action)
        """
        available = [a for a in self.env.get_available_actions() if a.enabled]
        
        if not available:
            return None
        
        state_hash = state.get_state_hash()
        
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(available)
        
        # Exploitation: choose best action for this state
        best_action = None
        best_value = float('-inf')
        
        for action in available:
            value = self.q_table[state_hash][action.name].average
            
            # Add exploration bonus for rarely-tried actions
            count = self.q_table[state_hash][action.name].count
            exploration_bonus = 0.1 / (count + 1)
            value += exploration_bonus
            
            if value > best_value:
                best_value = value
                best_action = action
        
        # If no good option found, random
        if best_action is None:
            best_action = random.choice(available)
        
        return best_action
    
    def learn_from_experience(self, exp: Experience):
        """Update Q-values from experience."""
        state_hash = exp.state_hash
        action_name = exp.action_name
        next_state_hash = exp.next_state_hash
        reward = exp.reward
        
        # Get current Q-value
        current_q = self.q_table[state_hash][action_name].average
        
        # Get max Q-value for next state
        if exp.done:
            max_next_q = 0.0
        else:
            next_values = [av.average for av in self.q_table[next_state_hash].values()]
            max_next_q = max(next_values) if next_values else 0.0
        
        # Q-learning update
        target = reward + self.discount_factor * max_next_q
        new_value = current_q + self.learning_rate * (target - current_q)
        
        # Update Q-table
        self.q_table[state_hash][action_name].update(new_value)
        
        # Store experience
        self.experiences.append(exp)
        if len(self.experiences) > self.max_experiences:
            self.experiences.pop(0)
        
        # Update seed memory if available
        if self.seed:
            self._update_seed_memory(exp)
    
    def _update_seed_memory(self, exp: Experience):
        """Update seed's associative memory with game experience."""
        # Create association: state + action -> outcome
        state_action = f"game_state:{exp.state_hash[:8]}+{exp.action_name}"
        outcome = "positive" if exp.reward > 0 else ("negative" if exp.reward < 0 else "neutral")
        
        # Perceive this as an experience
        self.seed.perceive(f"{exp.action_name} resulted in {outcome}")
    
    def record_action_effect(self, action: Action, result: ActionResult):
        """Record what effect an action had."""
        # Describe the effect
        effects = []
        
        # Check text changes
        for key in result.state_after.visible_text:
            before = result.state_before.visible_text.get(key, "")
            after = result.state_after.visible_text.get(key, "")
            if before != after:
                effects.append(f"text_changed:{key}")
        
        # Check numeric changes
        for key in result.state_after.numeric_values:
            before = result.state_before.numeric_values.get(key, 0)
            after = result.state_after.numeric_values.get(key, 0)
            if after > before:
                effects.append(f"increased:{key}")
            elif after < before:
                effects.append(f"decreased:{key}")
        
        # Check button state changes
        for key in result.state_after.button_states:
            before = result.state_before.button_states.get(key, True)
            after = result.state_after.button_states.get(key, True)
            if before != after:
                effects.append(f"button_{'enabled' if after else 'disabled'}:{key}")
        
        if result.done:
            effects.append("episode_ended")
        
        # Record effects
        effect_key = "|".join(sorted(effects)) or "no_effect"
        self.action_effects[action.name][effect_key] += 1
    
    def step(self) -> Optional[ActionResult]:
        """Take one step in the environment."""
        state = self.env.observe()
        action = self.select_action(state)
        
        if action is None:
            return None
        
        # Execute action
        result = self.env.execute_action(action)
        
        # Record experience
        exp = Experience(
            state_hash=state.get_state_hash(),
            action_name=action.name,
            next_state_hash=result.state_after.get_state_hash(),
            reward=result.reward,
            done=result.done
        )
        
        # Learn from experience
        self.learn_from_experience(exp)
        
        # Record action effects
        self.record_action_effect(action, result)
        
        # Update statistics
        self.total_steps += 1
        self.total_reward += result.reward
        self.current_episode_reward += result.reward
        self.current_episode_steps += 1
        
        return result
    
    def run_episode(self, max_steps: int = 1000) -> float:
        """Run one complete episode."""
        self.env.reset()
        self.current_episode_reward = 0.0
        self.current_episode_steps = 0
        
        for _ in range(max_steps):
            result = self.step()
            
            if result is None or result.done:
                break
        
        # Update epsilon (decay exploration)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Record episode
        self.total_episodes += 1
        self.episode_rewards.append(self.current_episode_reward)
        
        return self.current_episode_reward
    
    def train(self, num_episodes: int, max_steps_per_episode: int = 1000,
              callback: callable = None) -> List[float]:
        """Train the agent for multiple episodes."""
        rewards = []
        
        for episode in range(num_episodes):
            reward = self.run_episode(max_steps_per_episode)
            rewards.append(reward)
            
            if callback:
                callback(episode, reward, self.epsilon)
        
        return rewards
    
    def get_best_action(self, state: Observation) -> Optional[Action]:
        """Get the best action for a state (no exploration)."""
        available = [a for a in self.env.get_available_actions() if a.enabled]
        
        if not available:
            return None
        
        state_hash = state.get_state_hash()
        
        best_action = None
        best_value = float('-inf')
        
        for action in available:
            value = self.q_table[state_hash][action.name].average
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action or random.choice(available)
    
    def play_episode(self, max_steps: int = 1000, delay: float = 0.2) -> float:
        """Play one episode using learned policy (no learning)."""
        self.env.reset()
        total_reward = 0.0
        
        for _ in range(max_steps):
            state = self.env.observe()
            action = self.get_best_action(state)
            
            if action is None:
                break
            
            result = self.env.execute_action(action)
            total_reward += result.reward
            
            if delay > 0:
                time.sleep(delay)
            
            if result.done:
                break
        
        return total_reward
    
    def get_action_summary(self) -> Dict[str, Dict]:
        """Get summary of learned action effects."""
        summary = {}
        
        for action_name, effects in self.action_effects.items():
            total = sum(effects.values())
            summary[action_name] = {
                "total_uses": total,
                "effects": {k: v/total for k, v in effects.items()} if total > 0 else {}
            }
        
        return summary
    
    def get_stats(self) -> Dict:
        """Get agent statistics."""
        recent_rewards = self.episode_rewards[-100:] if self.episode_rewards else [0]
        
        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "total_reward": self.total_reward,
            "epsilon": self.epsilon,
            "average_reward": sum(recent_rewards) / len(recent_rewards),
            "known_actions": len(self.known_actions),
            "states_visited": len(self.q_table),
            "experiences_stored": len(self.experiences)
        }


class MazeAgent(GameAgent):
    """Specialized agent for maze navigation."""
    
    def __init__(self, env: Environment, seed=None):
        super().__init__(env, seed)
        
        # Maze-specific: track position patterns
        self.movement_keys = ['key_up', 'key_down', 'key_left', 'key_right',
                              'key_w', 'key_s', 'key_a', 'key_d']
        
        # Shorter epsilon for faster learning
        self.epsilon_decay = 0.99
    
    def select_action(self, state: Observation) -> Optional[Action]:
        """Maze-specific action selection - prefer movement keys."""
        available = [a for a in self.env.get_available_actions() if a.enabled]
        
        if not available:
            return None
        
        # Filter to movement actions when in game
        movement_actions = [a for a in available if a.name in self.movement_keys]
        
        if movement_actions:
            available = movement_actions
        
        state_hash = state.get_state_hash()
        
        # Exploration
        if random.random() < self.epsilon:
            return random.choice(available)
        
        # Exploitation with maze heuristics
        best_action = None
        best_value = float('-inf')
        
        for action in available:
            value = self.q_table[state_hash][action.name].average
            
            # Penalize recently taken actions to avoid loops
            count = self.q_table[state_hash][action.name].count
            if count > 5:
                value -= 0.1 * math.log(count)
            
            # Exploration bonus
            value += 0.1 / (count + 1)
            
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action or random.choice(available)


class BlackjackAgent(GameAgent):
    """Specialized agent for blackjack."""
    
    def __init__(self, env: Environment, seed=None):
        super().__init__(env, seed)
        
        # Blackjack-specific knowledge
        self.action_priority = ['btn_hit', 'btn_stand', 'btn_double', 'btn_split']
        
        # Slower epsilon decay for card games
        self.epsilon_decay = 0.999
        self.epsilon_min = 0.1  # Keep some exploration
    
    def select_action(self, state: Observation) -> Optional[Action]:
        """Blackjack-specific action selection."""
        available = [a for a in self.env.get_available_actions() if a.enabled]
        
        if not available:
            return None
        
        # Check if we need to deal first
        deal_actions = [a for a in available if 'deal' in a.name.lower()]
        if deal_actions:
            # Need to place bet and deal
            # Set bet in entry first
            for name, entry in self.env.entries.items():
                try:
                    entry.delete(0, tk.END)
                    entry.insert(0, "10")
                except:
                    pass
            return deal_actions[0]
        
        # Filter to game actions
        game_actions = [a for a in available 
                       if any(p in a.name for p in self.action_priority)]
        
        if game_actions:
            available = game_actions
        
        state_hash = state.get_state_hash()
        
        # Extract hand value from state
        hand_value = self._extract_hand_value(state)
        
        # Simple strategy hints (before full learning kicks in)
        if self.total_episodes < 50 and hand_value is not None:
            # Basic strategy hints
            if hand_value >= 17:
                stand = [a for a in available if 'stand' in a.name.lower()]
                if stand:
                    return stand[0]
            elif hand_value <= 11:
                hit = [a for a in available if 'hit' in a.name.lower()]
                if hit:
                    return hit[0]
        
        # Standard epsilon-greedy after initial exploration
        if random.random() < self.epsilon:
            return random.choice(available)
        
        # Best learned action
        best_action = None
        best_value = float('-inf')
        
        for action in available:
            value = self.q_table[state_hash][action.name].average
            if value > best_value:
                best_value = value
                best_action = action
        
        return best_action or random.choice(available)
    
    def _extract_hand_value(self, state: Observation) -> Optional[int]:
        """Try to extract player's hand value from visible text."""
        for key, text in state.visible_text.items():
            if 'value' in key.lower() or 'value' in text.lower():
                # Look for number after "Value:"
                import re
                match = re.search(r'value[:\s]*(\d+)', text.lower())
                if match:
                    return int(match.group(1))
        return None


def create_agent(game_type: str, env: Environment, seed=None) -> GameAgent:
    """Factory function to create appropriate agent."""
    if game_type.lower() == "maze":
        return MazeAgent(env, seed)
    elif game_type.lower() == "blackjack":
        return BlackjackAgent(env, seed)
    else:
        return GameAgent(env, seed)
