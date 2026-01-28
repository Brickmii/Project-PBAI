"""
Action Decoders

Convert PBAI decisions to Gymnasium action space format.
Also provide semantic names for actions.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import math

logger = logging.getLogger(__name__)


class ActionDecoder(ABC):
    """Base class for action decoders."""
    
    def __init__(self, space, env_name: str = ""):
        self.space = space
        self.env_name = env_name
        self.action_names: List[str] = []
        self.n_actions: int = 0
    
    @abstractmethod
    def decode(self, action_idx: int) -> str:
        """Convert action index to semantic name."""
        pass
    
    @abstractmethod
    def encode(self, action_name: str) -> int:
        """Convert semantic name to action index."""
        pass


class DiscreteDecoder(ActionDecoder):
    """Decoder for Discrete action spaces."""
    
    # Known action names for common environments
    KNOWN_ACTIONS = {
        "CartPole": ["left", "right"],
        "MountainCar": ["left", "neutral", "right"],
        "Acrobot": ["neg_torque", "zero_torque", "pos_torque"],
        "LunarLander": ["noop", "left_engine", "main_engine", "right_engine"],
        "FrozenLake": ["left", "down", "right", "up"],
        "Taxi": ["south", "north", "east", "west", "pickup", "dropoff"],
        "Blackjack": ["stand", "hit"],
        "CliffWalking": ["up", "right", "down", "left"],
    }
    
    def __init__(self, space, env_name: str = ""):
        super().__init__(space, env_name)
        self.n_actions = space.n
        self.action_names = self._get_action_names()
    
    def _get_action_names(self) -> List[str]:
        """Get semantic action names."""
        for name, actions in self.KNOWN_ACTIONS.items():
            if name in self.env_name:
                if len(actions) == self.n_actions:
                    return actions
        
        # Default numbered actions
        return [f"action_{i}" for i in range(self.n_actions)]
    
    def decode(self, action_idx: int) -> str:
        """Convert index to name."""
        if 0 <= action_idx < len(self.action_names):
            return self.action_names[action_idx]
        return f"action_{action_idx}"
    
    def encode(self, action_name: str) -> int:
        """Convert name to index."""
        if action_name in self.action_names:
            return self.action_names.index(action_name)
        # Try parsing action_N format
        if action_name.startswith("action_"):
            try:
                return int(action_name.split("_")[1])
            except:
                pass
        raise ValueError(f"Unknown action: {action_name}")


class BoxDecoder(ActionDecoder):
    """
    Decoder for Box (continuous) action spaces.
    
    Discretizes continuous actions into bins.
    """
    
    def __init__(self, space, env_name: str = "", n_bins: int = 5):
        super().__init__(space, env_name)
        self.n_bins = n_bins
        self.low = space.low
        self.high = space.high
        self.shape = space.shape
        
        # Total discrete actions (bins per dimension)
        flat_size = 1
        for dim in self.shape:
            flat_size *= dim
        self.n_actions = n_bins ** flat_size
        
        # Generate action names
        self.action_names = self._generate_action_names()
    
    def _generate_action_names(self) -> List[str]:
        """Generate names for discretized actions."""
        labels = ["min", "low", "mid", "high", "max"]
        if self.n_bins != 5:
            labels = [f"b{i}" for i in range(self.n_bins)]
        
        # For single dimension
        if self.shape == (1,) or len(self.shape) == 0:
            return labels[:self.n_bins]
        
        # For multi-dimension, just number them
        return [f"action_{i}" for i in range(self.n_actions)]
    
    def decode(self, action_idx: int) -> str:
        if 0 <= action_idx < len(self.action_names):
            return self.action_names[action_idx]
        return f"action_{action_idx}"
    
    def encode(self, action_name: str) -> int:
        if action_name in self.action_names:
            return self.action_names.index(action_name)
        if action_name.startswith("action_"):
            try:
                return int(action_name.split("_")[1])
            except:
                pass
        raise ValueError(f"Unknown action: {action_name}")
    
    def to_continuous(self, action_idx: int):
        """Convert discrete action index to continuous action."""
        import numpy as np
        
        flat_size = 1
        for dim in self.shape:
            flat_size *= dim
        
        # Decode index to bin indices
        bin_indices = []
        remaining = action_idx
        for _ in range(flat_size):
            bin_indices.append(remaining % self.n_bins)
            remaining //= self.n_bins
        
        # Convert bins to continuous values
        continuous = []
        for i, bin_idx in enumerate(bin_indices):
            low = self.low.flatten()[i] if hasattr(self.low, 'flatten') else self.low
            high = self.high.flatten()[i] if hasattr(self.high, 'flatten') else self.high
            
            # Map bin to value
            t = bin_idx / (self.n_bins - 1) if self.n_bins > 1 else 0.5
            value = low + t * (high - low)
            continuous.append(value)
        
        return np.array(continuous).reshape(self.shape)


class MultiDiscreteDecoder(ActionDecoder):
    """Decoder for MultiDiscrete action spaces."""
    
    def __init__(self, space, env_name: str = ""):
        super().__init__(space, env_name)
        self.nvec = space.nvec
        
        # Total actions = product of dimensions
        self.n_actions = 1
        for n in self.nvec:
            self.n_actions *= n
        
        self.action_names = [f"action_{i}" for i in range(self.n_actions)]
    
    def decode(self, action_idx: int) -> str:
        return f"action_{action_idx}"
    
    def encode(self, action_name: str) -> int:
        if action_name.startswith("action_"):
            return int(action_name.split("_")[1])
        raise ValueError(f"Unknown action: {action_name}")


def create_decoder(space, env_name: str = "") -> ActionDecoder:
    """
    Create appropriate decoder for a Gym action space.
    
    Args:
        space: Gymnasium action space
        env_name: Environment name for context
        
    Returns:
        Appropriate decoder instance
    """
    space_type = type(space).__name__
    
    if space_type == "Discrete":
        return DiscreteDecoder(space, env_name)
    elif space_type == "Box":
        return BoxDecoder(space, env_name)
    elif space_type == "MultiDiscrete":
        return MultiDiscreteDecoder(space, env_name)
    else:
        logger.warning(f"Unknown action space type: {space_type}, using discrete")
        return DiscreteDecoder(space, env_name)
