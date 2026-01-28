#!/usr/bin/env python3
"""
PBAI Thermal Manifold - Driver Runner

Run any environment by specifying its driver.

Usage:
    python -m drivers.run <driver_name> [options]
    
    # Or from project root:
    python drivers/run.py <driver_name> [options]
    
Examples:
    python drivers/run.py mock
    python drivers/run.py mock --loops 100
    python drivers/run.py minecraft --config '{"host": "localhost", "port": 25565}'
    
Available drivers are any *_driver.py files in this directory.
"""

import sys
import os
import json
import argparse
import logging
from time import time, sleep

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Manifold, search, K
from drivers import EnvironmentCore, DriverLoader, MockDriver

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def list_drivers() -> list:
    """List available drivers."""
    loader = DriverLoader()
    return loader.discover_drivers()


def run_environment(
    driver_name: str,
    config: dict = None,
    max_loops: int = None,
    loop_delay: float = 0.1,
    verbose: bool = False
) -> None:
    """
    Run PBAI with a specific environment driver.
    
    Args:
        driver_name: Name of driver (e.g., "mock", "minecraft")
        config: Configuration dict for the driver
        max_loops: Max iterations (None = infinite)
        loop_delay: Seconds between loops
        verbose: Show debug output
    """
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # ─────────────────────────────────────────────────────────────────────────
    # SETUP
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(f"Loading driver: {driver_name}")
    
    # Load driver
    loader = DriverLoader()
    
    if driver_name == "mock":
        driver = MockDriver(config)
    else:
        driver = loader.load_driver(driver_name, config)
    
    if not driver:
        logger.error(f"Failed to load driver: {driver_name}")
        logger.info(f"Available drivers: {list_drivers()}")
        return
    
    # Create environment core
    core = EnvironmentCore()
    core.register_driver(driver)
    core.activate_driver(driver.DRIVER_ID)
    
    # Create manifold
    manifold = Manifold()
    manifold.bootstrap()
    
    logger.info(f"Driver '{driver.DRIVER_NAME}' activated")
    logger.info(f"Manifold bootstrapped with {len(manifold.nodes)} nodes")
    logger.info("Starting perception-action loop... (Ctrl+C to stop)")
    
    # ─────────────────────────────────────────────────────────────────────────
    # MAIN LOOP
    # ─────────────────────────────────────────────────────────────────────────
    loop_count = 0
    
    try:
        while max_loops is None or loop_count < max_loops:
            loop_count += 1
            manifold.loop_number = loop_count
            
            # PERCEIVE
            perception = core.perceive()
            
            if perception:
                # Process entities through manifold
                for entity in perception.entities:
                    search(str(entity), manifold)
                
                # Process events
                for event in perception.events:
                    search(str(event), manifold)
            
            # THINK - find focus (hottest node)
            hot_nodes = sorted(
                [n for n in manifold.nodes.values() if n.concept != "self"],
                key=lambda n: n.heat,
                reverse=True
            )
            focus = hot_nodes[0] if hot_nodes else None
            
            # ACT - if sufficient heat
            if focus and focus.heat > K * 2:
                from drivers import Action
                action = Action(
                    action_type="observe",
                    target=focus.concept,
                    parameters={"reason": "high_heat"}
                )
                
                result = core.act(action)
                
                if result and result.success:
                    focus.spend_heat(K)
                    logger.debug(f"Loop {loop_count}: Acted on {focus.concept}")
            
            # STATUS
            if loop_count % 10 == 0:
                logger.info(f"Loop {loop_count}: {len(manifold.nodes)} nodes, "
                           f"focus={focus.concept if focus else 'none'}")
            
            # SAVE
            if loop_count % 100 == 0:
                manifold.save_growth_map()
            
            sleep(loop_delay)
            
    except KeyboardInterrupt:
        logger.info("\nStopped by user")
    
    # ─────────────────────────────────────────────────────────────────────────
    # CLEANUP
    # ─────────────────────────────────────────────────────────────────────────
    logger.info(f"Ran {loop_count} loops")
    logger.info(f"Final node count: {len(manifold.nodes)}")
    
    manifold.save_growth_map()
    core.deactivate_driver()


def main():
    parser = argparse.ArgumentParser(
        description="Run PBAI with an environment driver",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python drivers/run.py mock
    python drivers/run.py mock --loops 50 --verbose
    python drivers/run.py minecraft --config '{"host": "localhost"}'
        """
    )
    
    parser.add_argument(
        "driver",
        nargs="?",
        default="mock",
        help="Driver name (default: mock)"
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="{}",
        help="JSON config string for driver"
    )
    parser.add_argument(
        "--loops", "-l",
        type=int,
        default=None,
        help="Max loops (default: infinite)"
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=0.1,
        help="Delay between loops in seconds (default: 0.1)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show debug output"
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available drivers and exit"
    )
    
    args = parser.parse_args()
    
    if args.list:
        drivers = list_drivers()
        print("Available drivers:")
        for d in drivers:
            print(f"  - {d}")
        print("\nBuilt-in: mock")
        return
    
    try:
        config = json.loads(args.config)
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON config: {e}")
        return
    
    run_environment(
        driver_name=args.driver,
        config=config,
        max_loops=args.loops,
        loop_delay=args.delay,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
