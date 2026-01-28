"""
PBAI Daemon - Raspberry Pi 5 Deployment

The living PBAI system. Runs continuously, choosing between environments,
learning, resting - all regulated by physical thermal constraints.

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────┐
    │                     PBAI DAEMON                         │
    │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐ │
    │  │  Tick Loop  │  │   Chooser   │  │  Network API    │ │
    │  │  (Clock)    │  │             │  │  (Remote Ctrl)  │ │
    │  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘ │
    │         │                │                   │          │
    │  ┌──────┴────────────────┴───────────────────┴───────┐ │
    │  │                    MANIFOLD                        │ │
    │  │   Identity ←→ Ego ←→ Conscience ←→ Nodes          │ │
    │  └───────────────────────┬───────────────────────────┘ │
    │                          │                              │
    │  ┌───────────────────────┴───────────────────────────┐ │
    │  │              ENVIRONMENT CORE                      │ │
    │  │  ┌──────┐ ┌──────┐ ┌──────┐ ┌──────┐ ┌──────────┐ │ │
    │  │  │ Maze │ │ BJ   │ │ Chat │ │Sensor│ │ Minecraft│ │ │
    │  │  └──────┘ └──────┘ └──────┘ └──────┘ └──────────┘ │ │
    │  └───────────────────────────────────────────────────┘ │
    │                          │                              │
    │  ┌───────────────────────┴───────────────────────────┐ │
    │  │              THERMAL MANAGER                       │ │
    │  │         CPU Temp → Tick Rate Regulation            │ │
    │  └───────────────────────────────────────────────────┘ │
    └─────────────────────────────────────────────────────────┘
              │              │              │
         [NVMe]       [Network]      [Sensors/GPIO]
"""

import logging
import threading
import time
import signal
import sys
import os
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Manifold, K, create_clock, Clock
from core.node_constants import (
    TICK_INTERVAL_BASE, TICK_INTERVAL_MIN, TICK_INTERVAL_MAX,
    PSYCHOLOGY_MIN_HEAT, COST_ACTION
)
from drivers import EnvironmentCore, Driver, Action, ActionResult

from .thermal import ThermalManager, ThermalState, create_thermal_manager

logger = logging.getLogger(__name__)


class DaemonState(Enum):
    """States of the PBAI daemon."""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"           # Manually paused
    THERMAL_PAUSE = "thermal"   # Paused due to heat
    RESTING = "resting"         # Voluntarily resting (low heat)
    STOPPING = "stopping"


@dataclass
class EnvironmentStats:
    """Statistics for an environment."""
    name: str
    sessions: int = 0
    total_heat_earned: float = 0.0
    total_heat_spent: float = 0.0
    successes: int = 0
    failures: int = 0
    last_used: Optional[datetime] = None
    
    @property
    def success_rate(self) -> float:
        total = self.successes + self.failures
        return self.successes / total if total > 0 else 0.5
    
    @property
    def net_heat(self) -> float:
        return self.total_heat_earned - self.total_heat_spent


@dataclass
class DaemonStats:
    """Statistics for the daemon."""
    started_at: Optional[datetime] = None
    total_ticks: int = 0
    total_choices: int = 0
    environment_stats: Dict[str, EnvironmentStats] = field(default_factory=dict)
    thermal_pauses: int = 0
    voluntary_rests: int = 0
    current_temp: float = 0.0
    current_zone: str = "unknown"


class EnvironmentChooser:
    """
    Chooses which environment to engage with based on psychology state.
    
    CHOICE FACTORS:
        - Ego heat (high = exploit known good environments)
        - Identity heat (high = explore uncertain environments)
        - Environment success history
        - Time since last use (novelty)
        - Available heat budget
    """
    
    def __init__(self, manifold: Manifold, env_core: EnvironmentCore):
        self.manifold = manifold
        self.env_core = env_core
        self.stats: Dict[str, EnvironmentStats] = {}
        
        # Rest is always available
        self.stats['rest'] = EnvironmentStats(name='rest')
    
    def register_environment(self, driver_id: str, name: str = None):
        """Register an environment for choice tracking."""
        name = name or driver_id
        if driver_id not in self.stats:
            self.stats[driver_id] = EnvironmentStats(name=name)
    
    def choose(self) -> str:
        """
        Choose the next environment to engage with.
        
        Returns:
            driver_id of chosen environment (or 'rest')
        """
        # Get psychology state
        ego_heat = self.manifold.ego_node.heat if self.manifold.ego_node else 0
        identity_heat = self.manifold.identity_node.heat if self.manifold.identity_node else 0
        total_heat = ego_heat + identity_heat
        
        # If too exhausted, must rest
        if total_heat < COST_ACTION:
            logger.info("Exhausted - choosing rest")
            return 'rest'
        
        # Calculate exploration rate (Identity / Total)
        exploration_rate = identity_heat / total_heat if total_heat > 0 else 0.5
        
        # Get available environments (registered drivers)
        available = [d for d in self.env_core.drivers.keys() if d != 'rest']
        
        if not available:
            return 'rest'
        
        # Score each environment
        scores = {}
        for env_id in available:
            stats = self.stats.get(env_id, EnvironmentStats(name=env_id))
            
            # Base score from success rate
            success_score = stats.success_rate
            
            # Novelty bonus (time since last use)
            novelty = 1.0
            if stats.last_used:
                hours_since = (datetime.now() - stats.last_used).total_seconds() / 3600
                novelty = min(2.0, 1.0 + hours_since * 0.1)
            
            # Net heat efficiency
            efficiency = 1.0
            if stats.sessions > 0:
                efficiency = max(0.5, 1.0 + stats.net_heat / (stats.sessions * K))
            
            # Combine based on exploration/exploitation
            exploit_score = success_score * efficiency
            explore_score = novelty * (1.0 - success_score + 0.5)  # Uncertainty bonus
            
            scores[env_id] = (
                exploit_score * (1 - exploration_rate) +
                explore_score * exploration_rate
            )
        
        # Choose (weighted random or argmax)
        import random
        if random.random() < 0.1:  # 10% pure random for diversity
            chosen = random.choice(available)
        else:
            chosen = max(scores, key=scores.get)
        
        logger.info(f"Chose environment: {chosen} (scores: {scores})")
        return chosen
    
    def record_outcome(self, driver_id: str, heat_earned: float, 
                       heat_spent: float, success: bool):
        """Record outcome of an environment session."""
        if driver_id not in self.stats:
            self.stats[driver_id] = EnvironmentStats(name=driver_id)
        
        stats = self.stats[driver_id]
        stats.sessions += 1
        stats.total_heat_earned += heat_earned
        stats.total_heat_spent += heat_spent
        if success:
            stats.successes += 1
        else:
            stats.failures += 1
        stats.last_used = datetime.now()


class PBAIDaemon:
    """
    The main PBAI daemon.
    
    Runs continuously on the Pi, managing:
    - Tick loop (thermally regulated)
    - Environment choice
    - Network API
    - Persistence
    """
    
    def __init__(self, 
                 save_path: str = None,
                 enable_api: bool = True,
                 api_port: int = 8420,
                 simulated_thermal: bool = False):
        """
        Initialize the daemon.
        
        Args:
            save_path: Path for manifold persistence
            enable_api: Enable network API
            api_port: Port for network API
            simulated_thermal: Use simulated thermal (for testing)
        """
        self.save_path = save_path or os.path.expanduser("~/.pbai/growth_map.json")
        self.enable_api = enable_api
        self.api_port = api_port
        
        # Ensure save directory exists
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        
        # Core components (initialized in start())
        self.manifold: Optional[Manifold] = None
        self.clock: Optional[Clock] = None
        self.env_core: Optional[EnvironmentCore] = None
        self.chooser: Optional[EnvironmentChooser] = None
        self.thermal: Optional[ThermalManager] = None
        
        # State
        self.state = DaemonState.STOPPED
        self.stats = DaemonStats()
        self._simulated_thermal = simulated_thermal
        
        # Threading
        self._main_thread: Optional[threading.Thread] = None
        self._api_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        
        # Current activity
        self.current_environment: Optional[str] = None
        self.current_session_heat: float = 0.0
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        logger.info(f"Received signal {signum}, shutting down...")
        self.stop()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def start(self):
        """Start the daemon."""
        if self.state != DaemonState.STOPPED:
            logger.warning("Daemon already running")
            return
        
        self.state = DaemonState.STARTING
        logger.info("═══ PBAI DAEMON STARTING ═══")
        
        # Initialize manifold
        self._init_manifold()
        
        # Initialize thermal manager
        self.thermal = create_thermal_manager(simulated=self._simulated_thermal)
        
        # Initialize environment core
        self.env_core = EnvironmentCore(manifold=self.manifold)
        
        # Initialize chooser
        self.chooser = EnvironmentChooser(self.manifold, self.env_core)
        
        # Initialize clock (but don't start its internal loop - we manage ticks)
        self.clock = Clock(self.manifold, save_path=self.save_path)
        
        # Start main loop
        self._stop_event.clear()
        self._main_thread = threading.Thread(target=self._main_loop, daemon=True)
        self._main_thread.start()
        
        # Start API if enabled
        if self.enable_api:
            self._start_api()
        
        self.state = DaemonState.RUNNING
        self.stats.started_at = datetime.now()
        logger.info("═══ PBAI DAEMON RUNNING ═══")
    
    def stop(self):
        """Stop the daemon."""
        if self.state == DaemonState.STOPPED:
            return
        
        self.state = DaemonState.STOPPING
        logger.info("═══ PBAI DAEMON STOPPING ═══")
        
        # Signal threads to stop
        self._stop_event.set()
        
        # Wait for main thread
        if self._main_thread:
            self._main_thread.join(timeout=5.0)
        
        # Final save
        if self.manifold:
            self.manifold.save_growth_map(self.save_path)
            logger.info(f"Final save to {self.save_path}")
        
        # Cleanup thermal
        if self.thermal:
            self.thermal.cleanup()
        
        self.state = DaemonState.STOPPED
        logger.info("═══ PBAI DAEMON STOPPED ═══")
    
    def _init_manifold(self):
        """Initialize or load the manifold."""
        self.manifold = Manifold()
        
        # Try to load existing state
        if os.path.exists(self.save_path):
            try:
                self.manifold.load_growth_map(self.save_path)
                logger.info(f"Loaded manifold from {self.save_path}")
                logger.info(f"  Nodes: {len(self.manifold.nodes)}")
                logger.info(f"  Loop: {self.manifold.loop_number}")
                return
            except Exception as e:
                logger.warning(f"Failed to load manifold: {e}")
        
        # Fresh start
        self.manifold.birth()
        logger.info("Fresh manifold created")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN LOOP
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _main_loop(self):
        """
        Main daemon loop.
        
        Each iteration:
        1. Check thermal state
        2. Perform tick (if not paused)
        3. Choose/engage environment (if active)
        4. Sleep based on thermal-adjusted interval
        """
        logger.info("Main loop started")
        
        while not self._stop_event.is_set():
            try:
                # 1. Check thermal state
                thermal_state = self.thermal.update()
                self.stats.current_temp = thermal_state.temperature
                self.stats.current_zone = thermal_state.zone
                
                # Thermal pause?
                if thermal_state.zone == 'critical':
                    if self.state != DaemonState.THERMAL_PAUSE:
                        self.state = DaemonState.THERMAL_PAUSE
                        self.stats.thermal_pauses += 1
                        logger.warning(f"THERMAL PAUSE: {thermal_state.temperature:.1f}°C")
                    time.sleep(5.0)  # Wait for cooldown
                    continue
                elif self.state == DaemonState.THERMAL_PAUSE:
                    self.state = DaemonState.RUNNING
                    logger.info(f"Thermal pause ended: {thermal_state.temperature:.1f}°C")
                
                # Manual pause?
                if self.state == DaemonState.PAUSED:
                    time.sleep(1.0)
                    continue
                
                # 2. Perform tick
                with self._lock:
                    self.clock._perform_tick()
                    self.stats.total_ticks += 1
                
                # 3. Environment activity (every N ticks)
                if self.stats.total_ticks % 10 == 0:
                    self._environment_cycle()
                
                # 4. Sleep (thermally adjusted)
                base_interval = self._calculate_interval()
                adjusted_interval = base_interval * thermal_state.tick_multiplier
                adjusted_interval = max(TICK_INTERVAL_MIN, 
                                       min(TICK_INTERVAL_MAX * 4, adjusted_interval))
                
                time.sleep(adjusted_interval)
                
            except Exception as e:
                logger.error(f"Main loop error: {e}")
                time.sleep(1.0)
        
        logger.info("Main loop ended")
    
    def _calculate_interval(self) -> float:
        """Calculate base tick interval from manifold heat."""
        return self.clock._calculate_interval()
    
    def _environment_cycle(self):
        """
        One environment engagement cycle.
        
        - Choose environment
        - Engage (perceive/act)
        - Record outcome
        """
        # Choose
        chosen = self.chooser.choose()
        self.stats.total_choices += 1
        
        if chosen == 'rest':
            # Voluntary rest
            self.state = DaemonState.RESTING
            self.stats.voluntary_rests += 1
            self.current_environment = None
            return
        
        self.state = DaemonState.RUNNING
        self.current_environment = chosen
        
        # Activate environment
        if chosen in self.env_core.drivers:
            self.env_core.activate_driver(chosen)
        else:
            logger.warning(f"Environment {chosen} not registered")
            return
        
        # Simple engagement: perceive → act
        heat_before = self.manifold.total_heat()
        
        try:
            perception = self.env_core.perceive()
            
            # Decide action (simple for now - just observe)
            # Real implementation would use manifold inference
            action = Action(action_type="observe")
            result = self.env_core.act(action)
            
            heat_after = self.manifold.total_heat()
            heat_delta = heat_after - heat_before
            
            # Record
            self.chooser.record_outcome(
                chosen,
                heat_earned=result.heat_value,
                heat_spent=COST_ACTION,
                success=result.success
            )
            
        except Exception as e:
            logger.error(f"Environment cycle error: {e}")
            self.chooser.record_outcome(chosen, 0, COST_ACTION, False)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # API (Network interface)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _start_api(self):
        """Start the network API."""
        try:
            from .api import create_api_server, run_api_server
            self._api_thread = threading.Thread(
                target=run_api_server,
                args=(self, self.api_port),
                daemon=True
            )
            self._api_thread.start()
            logger.info(f"API server started on port {self.api_port}")
        except ImportError:
            logger.warning("API module not available")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PUBLIC INTERFACE (for API and direct control)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def pause(self):
        """Pause the daemon."""
        if self.state == DaemonState.RUNNING:
            self.state = DaemonState.PAUSED
            logger.info("Daemon paused")
    
    def resume(self):
        """Resume the daemon."""
        if self.state == DaemonState.PAUSED:
            self.state = DaemonState.RUNNING
            logger.info("Daemon resumed")
    
    def force_save(self):
        """Force immediate save."""
        with self._lock:
            self.clock.force_save()
    
    def get_status(self) -> dict:
        """Get current daemon status."""
        return {
            "state": self.state.value,
            "uptime": (datetime.now() - self.stats.started_at).total_seconds() 
                      if self.stats.started_at else 0,
            "ticks": self.stats.total_ticks,
            "choices": self.stats.total_choices,
            "temperature": self.stats.current_temp,
            "thermal_zone": self.stats.current_zone,
            "current_environment": self.current_environment,
            "manifold": {
                "nodes": len(self.manifold.nodes) if self.manifold else 0,
                "loop": self.manifold.loop_number if self.manifold else 0,
                "total_heat": self.manifold.total_heat() if self.manifold else 0,
            },
            "psychology": {
                "identity": self.manifold.identity_node.heat if self.manifold and self.manifold.identity_node else 0,
                "ego": self.manifold.ego_node.heat if self.manifold and self.manifold.ego_node else 0,
                "conscience": self.manifold.conscience_node.heat if self.manifold and self.manifold.conscience_node else 0,
            },
            "environments": {
                k: {
                    "sessions": v.sessions,
                    "success_rate": v.success_rate,
                    "net_heat": v.net_heat
                }
                for k, v in self.chooser.stats.items()
            } if self.chooser else {}
        }
    
    def register_driver(self, driver: Driver, name: str = None):
        """Register a new environment driver."""
        with self._lock:
            self.env_core.register_driver(driver)
            self.chooser.register_environment(driver.DRIVER_ID, name)
            logger.info(f"Registered environment: {driver.DRIVER_ID}")
    
    def inject_perception(self, perception: dict):
        """Inject a perception from external source."""
        from drivers import Perception
        p = Perception(**perception)
        if self.env_core:
            self.env_core._integrate_perception(p)
    
    def inject_heat(self, amount: float, target: str = "identity"):
        """Inject heat into psychology (for external rewards)."""
        with self._lock:
            if target == "identity" and self.manifold.identity_node:
                self.manifold.identity_node.add_heat_unchecked(amount)
            elif target == "ego" and self.manifold.ego_node:
                self.manifold.ego_node.add_heat_unchecked(amount)
            elif target == "conscience" and self.manifold.conscience_node:
                self.manifold.conscience_node.add_heat_unchecked(amount)


def run_daemon(save_path: str = None, api_port: int = 8420, simulated: bool = False):
    """
    Run the PBAI daemon.
    
    Args:
        save_path: Path for manifold persistence
        api_port: Port for network API
        simulated: Use simulated thermal (for testing)
    """
    daemon = PBAIDaemon(
        save_path=save_path,
        enable_api=True,
        api_port=api_port,
        simulated_thermal=simulated
    )
    
    daemon.start()
    
    # Keep running until stopped
    try:
        while daemon.state != DaemonState.STOPPED:
            time.sleep(1.0)
    except KeyboardInterrupt:
        pass
    finally:
        daemon.stop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        datefmt='%H:%M:%S'
    )
    
    import argparse
    parser = argparse.ArgumentParser(description="PBAI Daemon")
    parser.add_argument("--save-path", default=None, help="Path for manifold persistence")
    parser.add_argument("--port", type=int, default=8420, help="API port")
    parser.add_argument("--simulated", action="store_true", help="Use simulated thermal")
    args = parser.parse_args()
    
    run_daemon(
        save_path=args.save_path,
        api_port=args.port,
        simulated=args.simulated
    )
