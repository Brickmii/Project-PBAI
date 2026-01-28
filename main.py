#!/usr/bin/env python3
"""
PBAI Thermal Manifold - Main Entry Point

PBAI is the agent. It experiences the environment, builds structure, and makes choices.

Architecture:
    Manifold <---> EnvironmentCore <---> Driver <---> External Environment

The Core Loop:
1. Perceive (auto-integrates into manifold)
2. Decide (collapse/correlate/select)
3. Act (execute motor action)
4. Learn (update Orders based on outcome)
5. Save growth map
"""

import os
import sys
import logging
import argparse
from time import time
from typing import Dict, Any, Optional

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import (
    Manifold, create_manifold, K,
    search, find,
    assert_self_valid,
    run_compression_tests,
    # Environment Core
    EnvironmentCore, create_environment_core, Perception, Action, ActionResult,
)
from core.node_constants import get_growth_path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(name)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("PBAI")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN LOOP
# ═══════════════════════════════════════════════════════════════════════════════

def main_loop(env_core: EnvironmentCore, max_loops: int = 10, verbose: bool = False):
    """
    Main PBAI loop.
    
    1. Perceive
    2. Decide (via collapse/correlate/select)
    3. Act
    4. Learn from outcome
    """
    manifold = env_core.manifold
    
    for loop in range(max_loops):
        logger.info(f"\n{'='*60}")
        logger.info(f"LOOP {loop + 1}/{max_loops}")
        logger.info(f"{'='*60}")
        
        # ─────────────────────────────────────────────────────────────────────
        # 1. PERCEIVE
        # ─────────────────────────────────────────────────────────────────────
        perception = env_core.perceive()
        state_key = perception.properties.get("state_key", "unknown")
        logger.info(f"Perceived: {state_key}")
        
        if verbose:
            logger.info(f"  Entities: {perception.entities}")
            logger.info(f"  Events: {perception.events}")
        
        # ─────────────────────────────────────────────────────────────────────
        # 2. DECIDE
        # ─────────────────────────────────────────────────────────────────────
        # Search for the state in manifold - this will correlate related concepts
        result = search(state_key, manifold)
        
        if result.action != "not_found":
            logger.info(f"Found state cluster: {len(result.cluster.get('all', []))} nodes")
        
        # For now, just take a random action from available
        # Future: use collapse/correlate/select on action nodes
        action = Action(
            type="interact",
            target=state_key,
            heat_cost=K * 0.1
        )
        
        logger.info(f"Decision: {action.type} → {action.target}")
        
        # ─────────────────────────────────────────────────────────────────────
        # 3. ACT
        # ─────────────────────────────────────────────────────────────────────
        result = env_core.act(action)
        
        logger.info(f"Result: success={result.success}, outcome={result.outcome}")
        
        if verbose:
            logger.info(f"  Heat value: {result.heat_value:.3f}")
        
        # ─────────────────────────────────────────────────────────────────────
        # 4. LEARN
        # ─────────────────────────────────────────────────────────────────────
        # Learning happens through Order updates on axes
        # This is handled by the driver node automatically
        
        # ─────────────────────────────────────────────────────────────────────
        # 5. SAVE (periodic)
        # ─────────────────────────────────────────────────────────────────────
        if loop % 5 == 4:
            manifold.save_growth_map()
            logger.info("Growth map saved")
    
    # Final save
    manifold.save_growth_map()
    logger.info("Final growth map saved")


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINTS
# ═══════════════════════════════════════════════════════════════════════════════

def run_demo():
    """Run a simple demo."""
    logger.info("=" * 60)
    logger.info("PBAI DEMO")
    logger.info("=" * 60)
    
    # Create manifold
    manifold = create_manifold()
    logger.info(f"Manifold created: {len(manifold.nodes)} nodes")
    
    # Validate Self
    assert_self_valid(manifold.self_node)
    logger.info("Self node valid ✓")
    
    # Create environment
    env_core = create_environment_core(manifold=manifold, use_mock=True)
    logger.info(f"Environment: {env_core.active_driver}")
    
    # Run a few loops
    main_loop(env_core, max_loops=5, verbose=True)
    
    # Show final state
    print("\n" + manifold.visualize())


def run_tests():
    """Run all tests."""
    from core import reset_birth_for_testing
    logger.info("Running tests...")
    
    # Test compression
    assert run_compression_tests(), "Compression tests failed!"
    logger.info("  Compression: PASSED ✓")
    
    # Test manifold
    reset_birth_for_testing()
    m = Manifold()
    m.birth()
    assert_self_valid(m.self_node)
    logger.info("  Manifold birth: PASSED ✓")
    
    # Test search
    result = search("test", m)
    assert result.action in ("found", "created")
    logger.info("  Search: PASSED ✓")
    
    logger.info("\nAll tests passed!")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="PBAI Thermal Manifold")
    parser.add_argument("--test", action="store_true", help="Run tests")
    parser.add_argument("--demo", action="store_true", help="Run demo")
    parser.add_argument("--loops", type=int, default=10, help="Number of loops")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--load", type=str, help="Load manifold from file")
    parser.add_argument("--driver", type=str, default="mock", help="Driver to use")
    
    args = parser.parse_args()
    
    if args.test:
        run_tests()
        return
    
    if args.demo:
        run_demo()
        return
    
    # Default: run main loop
    logger.info("=" * 60)
    logger.info("PBAI THERMAL MANIFOLD")
    logger.info("=" * 60)
    
    # Create or load manifold
    if args.load and os.path.exists(args.load):
        logger.info(f"Loading manifold from {args.load}")
        manifold = create_manifold(args.load)
    else:
        logger.info("Creating new manifold")
        manifold = create_manifold()
    
    # Create environment
    env_core = create_environment_core(
        manifold=manifold,
        use_mock=(args.driver == "mock")
    )
    logger.info(f"Environment: {env_core.active_driver}")
    
    # Run main loop
    try:
        main_loop(env_core, max_loops=args.loops, verbose=args.verbose)
    except KeyboardInterrupt:
        logger.info("\nInterrupted by user")
        manifold.save_growth_map()
    finally:
        env_core.deactivate_driver()
    
    # Final visualization
    print("\n" + manifold.visualize())


if __name__ == "__main__":
    main()
