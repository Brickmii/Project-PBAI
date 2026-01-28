"""
Observation Encoders

Convert Gymnasium observations into manifold-compatible representations.

CRITICAL: Good encoding = good learning. The manifold can only generalize
across states that LOOK similar. If "taxi at row 2" and "taxi at row 3"
are encoded as unrelated integers, no generalization happens.

Encoding strategies:
- Grid games: Encode as row/col so spatial patterns emerge
- Tuple games: Extract semantic components (Blackjack: soft/hard, sum, dealer)
- Box games: Bin continuous values with meaningful labels
- Complex games: Decompose state into learnable components
"""

import logging
import math
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple, Optional

logger = logging.getLogger(__name__)


class ObservationEncoder(ABC):
    """Base class for observation encoders."""
    
    def __init__(self, space, env_name: str = ""):
        self.space = space
        self.env_name = env_name
    
    @abstractmethod
    def encode_key(self, observation) -> str:
        """Encode observation to a string key for node lookup."""
        pass
    
    @abstractmethod
    def encode_features(self, observation) -> Dict[str, float]:
        """Encode observation to feature dict for node creation."""
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# GRID-BASED ENCODERS (FrozenLake, CliffWalking, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

class GridEncoder(ObservationEncoder):
    """
    Encoder for grid-based environments.
    
    Converts state integer to row/col coordinates so the manifold
    can learn spatial patterns like "moving down from row 2..."
    """
    
    def __init__(self, space, env_name: str = "", n_cols: int = 4, n_rows: int = 4):
        super().__init__(space, env_name)
        self.n_cols = n_cols
        self.n_rows = n_rows
    
    def encode_key(self, observation) -> str:
        state = int(observation)
        row = state // self.n_cols
        col = state % self.n_cols
        return f"{self.env_name}_r{row}c{col}"
    
    def encode_features(self, observation) -> Dict[str, float]:
        state = int(observation)
        row = state // self.n_cols
        col = state % self.n_cols
        return {
            "row": float(row),
            "col": float(col),
            "row_norm": row / max(1, self.n_rows - 1),
            "col_norm": col / max(1, self.n_cols - 1)
        }


class FrozenLakeEncoder(GridEncoder):
    """
    Encoder for FrozenLake-v1 (4x4 grid).
    
    States 0-15 map to a 4x4 grid:
    - Start: (0,0)
    - Goal: (3,3)
    - Holes: Various positions
    """
    
    def __init__(self, space, env_name: str = ""):
        # Determine grid size from env name
        if "8x8" in env_name:
            super().__init__(space, env_name, n_cols=8, n_rows=8)
        else:
            super().__init__(space, env_name, n_cols=4, n_rows=4)
    
    def encode_key(self, observation) -> str:
        state = int(observation)
        row = state // self.n_cols
        col = state % self.n_cols
        
        # Add semantic hints for special positions
        if state == 0:
            return f"{self.env_name}_START_r{row}c{col}"
        elif state == self.n_rows * self.n_cols - 1:
            return f"{self.env_name}_GOAL_r{row}c{col}"
        else:
            return f"{self.env_name}_r{row}c{col}"


class CliffWalkingEncoder(GridEncoder):
    """
    Encoder for CliffWalking-v1 (4x12 grid).
    
    States 0-47 map to a 4 row x 12 col grid:
    - Start: (3,0) bottom-left
    - Goal: (3,11) bottom-right
    - Cliff: (3,1) through (3,10) - instant death
    """
    
    def __init__(self, space, env_name: str = ""):
        super().__init__(space, env_name, n_cols=12, n_rows=4)
    
    def encode_key(self, observation) -> str:
        state = int(observation)
        row = state // self.n_cols
        col = state % self.n_cols
        
        # Add semantic hints
        if state == 36:  # Start position (row 3, col 0)
            return f"{self.env_name}_START_r{row}c{col}"
        elif state == 47:  # Goal position (row 3, col 11)
            return f"{self.env_name}_GOAL_r{row}c{col}"
        elif row == 3 and 1 <= col <= 10:  # Cliff
            return f"{self.env_name}_CLIFF_r{row}c{col}"
        else:
            return f"{self.env_name}_r{row}c{col}"
    
    def encode_features(self, observation) -> Dict[str, float]:
        features = super().encode_features(observation)
        state = int(observation)
        row = state // self.n_cols
        col = state % self.n_cols
        
        # Add cliff proximity warning
        features["near_cliff"] = 1.0 if row == 2 else 0.0
        features["on_cliff_row"] = 1.0 if row == 3 else 0.0
        features["dist_to_goal"] = abs(3 - row) + abs(11 - col)
        return features


class TaxiEncoder(ObservationEncoder):
    """
    Encoder for Taxi-v3.
    
    The state integer encodes 4 things:
    - taxi_row (0-4): 5 rows
    - taxi_col (0-4): 5 columns
    - passenger_loc (0-4): 0=R, 1=G, 2=Y, 3=B, 4=in_taxi
    - destination (0-3): 0=R, 1=G, 2=Y, 3=B
    
    State = ((taxi_row * 5 + taxi_col) * 5 + passenger_loc) * 4 + destination
    
    This encoder creates keys like:
    - "Taxi-v3_seeking_r2c3_toG" (passenger not picked up, going to G)
    - "Taxi-v3_carrying_r1c0_toB" (passenger in taxi, going to B)
    """
    
    LOCATIONS = ['R', 'G', 'Y', 'B']
    
    def __init__(self, space, env_name: str = ""):
        super().__init__(space, env_name)
    
    def _decode_state(self, state: int) -> Tuple[int, int, int, int]:
        """Decode state integer to components."""
        state = int(state)
        destination = state % 4
        state //= 4
        passenger_loc = state % 5
        state //= 5
        taxi_col = state % 5
        taxi_row = state // 5
        return taxi_row, taxi_col, passenger_loc, destination
    
    def encode_key(self, observation) -> str:
        taxi_row, taxi_col, passenger_loc, destination = self._decode_state(observation)
        
        dest_name = self.LOCATIONS[destination]
        
        if passenger_loc == 4:
            # Passenger in taxi - in delivery phase
            phase = "carrying"
        else:
            # Passenger waiting - in pickup phase
            phase = "seeking"
            # Could add passenger location but might make state space too sparse
        
        return f"{self.env_name}_{phase}_r{taxi_row}c{taxi_col}_to{dest_name}"
    
    def encode_features(self, observation) -> Dict[str, float]:
        taxi_row, taxi_col, passenger_loc, destination = self._decode_state(observation)
        
        dest_positions = {0: (0, 0), 1: (0, 4), 2: (4, 0), 3: (4, 3)}  # R, G, Y, B
        pass_positions = {0: (0, 0), 1: (0, 4), 2: (4, 0), 3: (4, 3), 4: None}
        
        has_passenger = passenger_loc == 4
        dest_row, dest_col = dest_positions[destination]
        
        features = {
            "taxi_row": float(taxi_row),
            "taxi_col": float(taxi_col),
            "has_passenger": 1.0 if has_passenger else 0.0,
            "destination": float(destination),
            "dist_to_dest": abs(taxi_row - dest_row) + abs(taxi_col - dest_col),
        }
        
        if not has_passenger:
            pass_row, pass_col = pass_positions[passenger_loc]
            features["dist_to_passenger"] = abs(taxi_row - pass_row) + abs(taxi_col - pass_col)
            features["at_passenger"] = 1.0 if features["dist_to_passenger"] == 0 else 0.0
        else:
            features["at_destination"] = 1.0 if features["dist_to_dest"] == 0 else 0.0
        
        return features


# ═══════════════════════════════════════════════════════════════════════════════
# GENERIC ENCODERS (Discrete, Box, Tuple, Dict)
# ═══════════════════════════════════════════════════════════════════════════════

class DiscreteEncoder(ObservationEncoder):
    """Generic encoder for Discrete observation spaces."""
    
    def encode_key(self, observation) -> str:
        if observation is None:
            return f"{self.env_name}_null_obs"
        try:
            return f"{self.env_name}_s{int(observation)}"
        except (TypeError, ValueError):
            return f"{self.env_name}_invalid_{hash(str(observation)) % 1000}"
    
    def encode_features(self, observation) -> Dict[str, float]:
        if observation is None:
            return {}
        try:
            return {f"state_{int(observation)}": 1.0}
        except (TypeError, ValueError):
            return {}


class BoxEncoder(ObservationEncoder):
    """
    Encoder for Box (continuous) observation spaces.
    
    Uses binning to discretize continuous values.
    Known environments get semantic feature names.
    """
    
    # Feature names for known environments
    KNOWN_FEATURES = {
        "CartPole": ["cart_pos", "cart_vel", "pole_angle", "pole_vel"],
        "MountainCar": ["position", "velocity"],
        "Acrobot": ["cos_t1", "sin_t1", "cos_t2", "sin_t2", "vel_t1", "vel_t2"],
        "LunarLander": ["x", "y", "vx", "vy", "angle", "angular_vel", "left_leg", "right_leg"],
        "Pendulum": ["cos_theta", "sin_theta", "theta_vel"],
    }
    
    def __init__(self, space, env_name: str = "", n_bins: int = 5):
        super().__init__(space, env_name)
        self.n_bins = n_bins
        self.low = space.low
        self.high = space.high
        self.shape = space.shape
        self.feature_names = self._get_feature_names()
    
    def _get_feature_names(self) -> List[str]:
        """Get meaningful feature names for known environments."""
        for name, features in self.KNOWN_FEATURES.items():
            if name in self.env_name:
                if len(features) == self._flat_size():
                    return features
        
        # Default: numbered features
        return [f"f{i}" for i in range(self._flat_size())]
    
    def _flat_size(self) -> int:
        """Get flattened observation size."""
        size = 1
        for dim in self.shape:
            size *= dim
        return size
    
    def _bin_value(self, value: float, low: float, high: float) -> int:
        """Bin a continuous value."""
        # Handle infinite bounds
        if math.isinf(low):
            low = -10.0
        if math.isinf(high):
            high = 10.0
        
        # Clip to bounds
        value = max(low, min(high, value))
        
        # Bin
        range_size = high - low
        if range_size == 0:
            return self.n_bins // 2
        
        normalized = (value - low) / range_size
        bin_idx = int(normalized * self.n_bins)
        return min(bin_idx, self.n_bins - 1)
    
    def _bin_label(self, bin_idx: int) -> str:
        """Get label for a bin."""
        if self.n_bins == 5:
            return ["vlo", "lo", "mid", "hi", "vhi"][bin_idx]
        elif self.n_bins == 3:
            return ["lo", "mid", "hi"][bin_idx]
        return str(bin_idx)
    
    def encode_key(self, observation) -> str:
        """Encode continuous observation to discrete key."""
        if observation is None:
            return f"{self.env_name}_null_obs"
        
        try:
            obs_flat = observation.flatten() if hasattr(observation, 'flatten') else [observation]
        except Exception:
            return f"{self.env_name}_invalid_obs"
        
        parts = []
        for i, value in enumerate(obs_flat):
            try:
                low = float(self.low.flatten()[i]) if hasattr(self.low, 'flatten') else float(self.low)
                high = float(self.high.flatten()[i]) if hasattr(self.high, 'flatten') else float(self.high)
                
                bin_idx = self._bin_value(float(value), low, high)
                name = self.feature_names[i] if i < len(self.feature_names) else f"f{i}"
                parts.append(f"{name[:3]}_{self._bin_label(bin_idx)}")
            except Exception:
                parts.append(f"f{i}_err")
        
        return f"{self.env_name}_{'_'.join(parts)}"
    
    def encode_features(self, observation) -> Dict[str, float]:
        """Encode to feature dict with normalized values."""
        if observation is None:
            return {}
        
        try:
            obs_flat = observation.flatten() if hasattr(observation, 'flatten') else [observation]
        except Exception:
            return {}
        
        features = {}
        for i, value in enumerate(obs_flat):
            try:
                name = self.feature_names[i] if i < len(self.feature_names) else f"f{i}"
                features[name] = float(value)
            except Exception:
                pass
        
        return features


class TupleEncoder(ObservationEncoder):
    """
    Encoder for Tuple observation spaces.
    
    Special handling for Blackjack-style tuples.
    DEFENSIVE: Handles None values gracefully.
    """
    
    def __init__(self, space, env_name: str = ""):
        super().__init__(space, env_name)
        self.sub_encoders = []
        
        for i, sub_space in enumerate(space.spaces):
            sub_encoder = create_encoder(sub_space, f"{env_name}_t{i}")
            self.sub_encoders.append(sub_encoder)
    
    def encode_key(self, observation) -> str:
        """Encode tuple to key."""
        # DEFENSIVE: Handle None observation
        if observation is None:
            return f"{self.env_name}_null_obs"
        
        # Special case: Blackjack
        if "Blackjack" in self.env_name:
            try:
                player_sum, dealer_card, usable_ace = observation
                # DEFENSIVE: Handle None values in observation
                if player_sum is None or dealer_card is None:
                    return f"{self.env_name}_invalid_state"
                soft = "s" if usable_ace else "h"
                return f"{self.env_name}_{soft}{player_sum}v{dealer_card}"
            except (ValueError, TypeError) as e:
                logger.warning(f"Blackjack encoding failed: {e}")
                return f"{self.env_name}_error_state"
        
        # Generic tuple encoding
        parts = []
        for obs_part in observation:
            if obs_part is None:
                parts.append("null")
            else:
                parts.append(str(obs_part))
        return f"{self.env_name}_s{'_'.join(parts)}"
    
    def encode_features(self, observation) -> Dict[str, float]:
        features = {}
        if observation is None:
            return features
        for i, (obs_part, encoder) in enumerate(zip(observation, self.sub_encoders)):
            if obs_part is None:
                continue
            try:
                sub_features = encoder.encode_features(obs_part)
                for k, v in sub_features.items():
                    features[f"t{i}_{k}"] = v
            except Exception:
                pass
        return features


class DictEncoder(ObservationEncoder):
    """Encoder for Dict observation spaces."""
    
    def __init__(self, space, env_name: str = ""):
        super().__init__(space, env_name)
        self.sub_encoders = {}
        
        for key, sub_space in space.spaces.items():
            self.sub_encoders[key] = create_encoder(sub_space, f"{env_name}_{key}")
    
    def encode_key(self, observation) -> str:
        parts = []
        for key in sorted(self.sub_encoders.keys()):
            if key in observation:
                sub_key = self.sub_encoders[key].encode_key(observation[key])
                # Extract just the state part
                parts.append(sub_key.split("_s")[-1] if "_s" in sub_key else sub_key)
        return f"{self.env_name}_s{'_'.join(parts)}"
    
    def encode_features(self, observation) -> Dict[str, float]:
        features = {}
        for key, encoder in self.sub_encoders.items():
            if key in observation:
                sub_features = encoder.encode_features(observation[key])
                for k, v in sub_features.items():
                    features[f"{key}_{k}"] = v
        return features


class MultiDiscreteEncoder(ObservationEncoder):
    """Encoder for MultiDiscrete observation spaces."""
    
    def encode_key(self, observation) -> str:
        parts = [str(int(v)) for v in observation]
        return f"{self.env_name}_s{'_'.join(parts)}"
    
    def encode_features(self, observation) -> Dict[str, float]:
        return {f"d{i}": float(v) for i, v in enumerate(observation)}


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_encoder(space, env_name: str = "") -> ObservationEncoder:
    """
    Create appropriate encoder for a Gym space.
    
    Special encoders for known environments, generic fallbacks otherwise.
    
    Args:
        space: Gymnasium observation space
        env_name: Environment name for context
        
    Returns:
        Appropriate encoder instance
    """
    space_type = type(space).__name__
    
    # ─────────────────────────────────────────────────────────────────────────
    # SPECIAL CASES: Environments that need semantic encoding
    # ─────────────────────────────────────────────────────────────────────────
    
    if "Taxi" in env_name:
        return TaxiEncoder(space, env_name)
    
    if "FrozenLake" in env_name:
        return FrozenLakeEncoder(space, env_name)
    
    if "CliffWalking" in env_name:
        return CliffWalkingEncoder(space, env_name)
    
    # ─────────────────────────────────────────────────────────────────────────
    # GENERIC: Fall back to space-type-based encoding
    # ─────────────────────────────────────────────────────────────────────────
    
    if space_type == "Discrete":
        return DiscreteEncoder(space, env_name)
    elif space_type == "Box":
        return BoxEncoder(space, env_name)
    elif space_type == "Tuple":
        return TupleEncoder(space, env_name)
    elif space_type == "Dict":
        return DictEncoder(space, env_name)
    elif space_type == "MultiDiscrete":
        return MultiDiscreteEncoder(space, env_name)
    else:
        logger.warning(f"Unknown space type: {space_type}, using generic discrete")
        return DiscreteEncoder(space, env_name)
