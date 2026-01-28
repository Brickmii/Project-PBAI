"""
Tests for PBAI Gym Driver

Run: python -m pytest tests/test_gym.py -v
Or: python tests/test_gym.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import reset_birth_for_testing

def test_encoders():
    """Test observation encoders."""
    from drivers.gym_adapters.encoders import DiscreteEncoder, BoxEncoder, create_encoder
    
    # Mock spaces
    class MockDiscrete:
        def __init__(self, n):
            self.n = n
    
    class MockBox:
        def __init__(self):
            import numpy as np
            self.low = np.array([-1.0, -1.0])
            self.high = np.array([1.0, 1.0])
            self.shape = (2,)
    
    # Test discrete
    space = MockDiscrete(16)
    encoder = DiscreteEncoder(space, "FrozenLake")
    key = encoder.encode_key(5)
    assert "5" in key
    print(f"✓ DiscreteEncoder: {key}")
    
    # Test box
    space = MockBox()
    encoder = BoxEncoder(space, "Test", n_bins=5)
    import numpy as np
    key = encoder.encode_key(np.array([0.0, 0.5]))
    assert "_" in key
    print(f"✓ BoxEncoder: {key}")
    
    print("Encoder tests passed!")


def test_decoders():
    """Test action decoders."""
    from drivers.gym_adapters.decoders import DiscreteDecoder, create_decoder
    
    class MockDiscrete:
        def __init__(self, n):
            self.n = n
    
    # Test FrozenLake actions
    space = MockDiscrete(4)
    decoder = DiscreteDecoder(space, "FrozenLake-v1")
    assert decoder.action_names == ["left", "down", "right", "up"]
    print(f"✓ FrozenLake actions: {decoder.action_names}")
    
    # Test encode/decode
    idx = decoder.encode("right")
    assert idx == 2
    name = decoder.decode(2)
    assert name == "right"
    print(f"✓ right -> {idx} -> {name}")
    
    # Test Blackjack
    space = MockDiscrete(2)
    decoder = DiscreteDecoder(space, "Blackjack-v1")
    assert decoder.action_names == ["stand", "hit"]
    print(f"✓ Blackjack actions: {decoder.action_names}")
    
    print("Decoder tests passed!")


def test_gym_driver():
    """Test GymDriver with game handlers - requires gymnasium."""
    try:
        import gymnasium as gym
    except ImportError:
        print("⚠ Gymnasium not installed, skipping GymDriver tests")
        return
    
    from core import Manifold
    from drivers.gym_driver import GymDriver, GridGameHandler, BlackjackGameHandler, GenericGameHandler
    
    reset_birth_for_testing()
    manifold = Manifold()
    manifold.bootstrap()
    
    # Test FrozenLake - should use GridGameHandler
    driver = GymDriver("FrozenLake-v1", manifold)
    driver.initialize()
    assert isinstance(driver.game_handler, GridGameHandler), f"Expected GridGameHandler, got {type(driver.game_handler)}"
    print(f"✓ FrozenLake uses GridGameHandler")
    
    # Test Blackjack - should use BlackjackGameHandler
    reset_birth_for_testing()
    manifold = Manifold()
    manifold.bootstrap()
    driver = GymDriver("Blackjack-v1", manifold)
    driver.initialize()
    assert isinstance(driver.game_handler, BlackjackGameHandler), f"Expected BlackjackGameHandler, got {type(driver.game_handler)}"
    print(f"✓ Blackjack uses BlackjackGameHandler")
    
    # Test CartPole - should use GenericGameHandler
    reset_birth_for_testing()
    manifold = Manifold()
    manifold.bootstrap()
    driver = GymDriver("CartPole-v1", manifold)
    driver.initialize()
    assert isinstance(driver.game_handler, GenericGameHandler), f"Expected GenericGameHandler, got {type(driver.game_handler)}"
    print(f"✓ CartPole uses GenericGameHandler")
    
    print("GymDriver tests passed!")


def test_with_gymnasium():
    """Test with actual gymnasium (if installed)."""
    try:
        import gymnasium as gym
    except ImportError:
        print("⚠ Gymnasium not installed, skipping gym tests")
        return
    
    from core import Manifold
    from drivers import create_gym_driver, EnvironmentCore
    
    # Test on Blackjack
    reset_birth_for_testing()
    manifold = Manifold()
    manifold.bootstrap()
    
    driver = create_gym_driver("Blackjack-v1", manifold=manifold)
    env_core = EnvironmentCore(manifold)
    env_core.register_driver(driver)
    env_core.activate_driver(driver.DRIVER_ID)
    
    # Run 10 episodes
    total_reward = 0
    episode = 0
    episode_reward = 0
    
    while episode < 10:
        action, result, heat = env_core.step()
        reward = result.changes.get("reward", 0)
        done = result.changes.get("done", False)
        episode_reward += reward
        
        if done:
            total_reward += episode_reward
            episode += 1
            driver.reset()
            episode_reward = 0
    
    avg_reward = total_reward / 10
    print(f"✓ Blackjack 10 episodes: avg reward = {avg_reward:.2f}")
    print("Gymnasium integration tests passed!")


if __name__ == "__main__":
    print("=" * 60)
    print("PBAI Gym Driver Tests")
    print("=" * 60)
    
    test_encoders()
    print()
    test_decoders()
    print()
    test_gym_driver()
    print()
    test_with_gymnasium()
    
    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)
