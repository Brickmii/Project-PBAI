"""
Gym Adapters - Encoders and Decoders for Gymnasium environments

Encoders convert observations to semantic state keys.
Decoders convert action names to gym action integers.
"""

from .encoders import (
    ObservationEncoder,
    DiscreteEncoder,
    BoxEncoder,
    TupleEncoder,
    DictEncoder,
    MultiDiscreteEncoder,
    GridEncoder,
    FrozenLakeEncoder,
    CliffWalkingEncoder,
    TaxiEncoder,
    create_encoder
)

from .decoders import (
    ActionDecoder,
    DiscreteDecoder,
    BoxDecoder,
    MultiDiscreteDecoder,
    create_decoder
)

__all__ = [
    # Encoders
    'ObservationEncoder', 'DiscreteEncoder', 'BoxEncoder',
    'TupleEncoder', 'DictEncoder', 'MultiDiscreteEncoder',
    'GridEncoder', 'FrozenLakeEncoder', 'CliffWalkingEncoder', 'TaxiEncoder',
    'create_encoder',
    
    # Decoders
    'ActionDecoder', 'DiscreteDecoder', 'BoxDecoder',
    'MultiDiscreteDecoder', 'create_decoder',
]
