"""
PBAI Gym Runner - Run gym environments through PBAI

Tests the flow:
    Gym Env → GymDriver → EnvironmentCore → Manifold → DecisionNode → back

Usage:
    python -m drivers.tasks.gym_runner --env CliffWalking-v1 --episodes 100
"""

import argparse
import logging
import sys
import os

# Setup path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core import get_pbai_manifold, K
from drivers.environment import EnvironmentCore
from drivers.gym_driver import GymDriver, create_gym_driver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def run_episodes(env_name: str, num_episodes: int = 100, render: bool = False):
    """
    Run episodes using PBAI architecture.
    
    Flow per step:
        1. EnvironmentCore.perceive() → GymDriver.perceive() → Perception
        2. EnvironmentCore.decide() → Psychology cycle → Action
        3. EnvironmentCore.act() → GymDriver.act() → ActionResult
        4. EnvironmentCore.feedback() → Psychology.outcome() → Heat changes
    """
    # Get the ONE PBAI manifold
    manifold = get_pbai_manifold()
    logger.info(f"PBAI manifold ready: {len(manifold.nodes)} nodes")
    
    # Create environment core (ENTRY point)
    env_core = EnvironmentCore(manifold)
    
    # Create gym driver
    render_mode = "human" if render else None
    driver = create_gym_driver(env_name, manifold=manifold, render_mode=render_mode)
    
    # Register and activate driver
    env_core.register_driver(driver)
    if not env_core.activate_driver(driver.DRIVER_ID):
        logger.error("Failed to activate driver")
        return
    
    logger.info(f"Running {num_episodes} episodes on {env_name}")
    logger.info(f"Available actions: {driver.get_available_actions()}")
    
    # Track stats
    episode_rewards = []
    best_reward = float('-inf')
    worst_reward = float('inf')
    
    for episode in range(num_episodes):
        episode_reward = 0.0
        step_count = 0
        done = False
        
        # Reset driver for new episode
        driver.reset()
        
        while not done:
            # Use the clean step() method
            action, result, heat_changes = env_core.step()
            
            episode_reward += result.changes.get("reward", 0)
            step_count += 1
            done = result.changes.get("done", False)
            
            # Prevent runaway episodes
            if step_count > 1000:
                logger.warning("Episode exceeded 1000 steps, forcing end")
                break
        
        episode_rewards.append(episode_reward)
        best_reward = max(best_reward, episode_reward)
        worst_reward = min(worst_reward, episode_reward)
        
        # Log every 10 episodes
        if (episode + 1) % 10 == 0:
            recent_avg = sum(episode_rewards[-10:]) / min(10, len(episode_rewards))
            logger.info(f"Episode {episode + 1}: reward={episode_reward:.1f}, "
                       f"recent_avg={recent_avg:.1f}")
    
    # Final stats
    avg_reward = sum(episode_rewards) / len(episode_rewards)
    logger.info("=" * 60)
    logger.info(f"RESULTS: {num_episodes} episodes on {env_name}")
    logger.info(f"  Average reward: {avg_reward:.2f}")
    logger.info(f"  Best reward: {best_reward:.2f}")
    logger.info(f"  Worst reward: {worst_reward:.2f}")
    logger.info(f"  Manifold nodes: {len(manifold.nodes)}")
    logger.info("=" * 60)
    
    # Shutdown
    env_core.deactivate_driver()
    
    return episode_rewards


def compare_baselines(env_name: str, num_episodes: int = 100):
    """Compare adaptive baseline vs random performance."""
    import random
    
    try:
        import gymnasium as gym
    except ImportError:
        logger.error("gymnasium not installed")
        return
    
    # Random baseline
    logger.info(f"Running random baseline on {env_name}...")
    env = gym.make(env_name)
    random_rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        while not done and steps < 1000:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            steps += 1
        random_rewards.append(episode_reward)
    
    env.close()
    random_avg = sum(random_rewards) / len(random_rewards)
    
    # PBAI with adaptive baseline
    logger.info(f"Running PBAI on {env_name}...")
    pbai_rewards = run_episodes(env_name, num_episodes)
    pbai_avg = sum(pbai_rewards) / len(pbai_rewards)
    
    # Compare
    logger.info("=" * 60)
    logger.info("COMPARISON")
    logger.info(f"  Random avg: {random_avg:.2f}")
    logger.info(f"  PBAI avg:   {pbai_avg:.2f}")
    logger.info(f"  Improvement: {((pbai_avg - random_avg) / abs(random_avg) * 100):.1f}%")
    logger.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="PBAI Gym Runner")
    parser.add_argument("--env", type=str, default="CliffWalking-v1",
                       help="Gymnasium environment name")
    parser.add_argument("--episodes", type=int, default=100,
                       help="Number of episodes to run")
    parser.add_argument("--render", action="store_true",
                       help="Render environment")
    parser.add_argument("--compare", action="store_true",
                       help="Compare to random baseline")
    
    args = parser.parse_args()
    
    if args.compare:
        compare_baselines(args.env, args.episodes)
    else:
        run_episodes(args.env, args.episodes, args.render)


if __name__ == "__main__":
    main()
