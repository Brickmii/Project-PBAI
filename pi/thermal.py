"""
PBAI Thermal Integration - Raspberry Pi 5

Maps physical CPU temperature to manifold tick rate.
The heat metaphor becomes literal - silicon heat = cognitive constraint.

EMBODIMENT:
    - Hot CPU → slower ticks (forced rest)
    - Cool CPU → faster ticks (can think more)
    - Thermal throttling = metabolic throttling
    
TEMPERATURE ZONES:
    < 50°C  : Cool    - Maximum tick rate
    50-65°C : Warm    - Normal tick rate  
    65-75°C : Hot     - Reduced tick rate
    > 75°C  : Danger  - Minimum tick rate (forced rest)
    > 80°C  : Critical - Pause entirely until cooled
"""

import logging
import os
from typing import Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Temperature zones (Celsius)
TEMP_COOL = 50.0      # Below this = maximum performance
TEMP_WARM = 65.0      # Normal operating range
TEMP_HOT = 75.0       # Start throttling
TEMP_DANGER = 80.0    # Severe throttling
TEMP_CRITICAL = 85.0  # Pause until cooled

# Tick interval multipliers for each zone
MULTIPLIER_COOL = 0.5     # 2x faster
MULTIPLIER_WARM = 1.0     # Normal
MULTIPLIER_HOT = 2.0      # 2x slower
MULTIPLIER_DANGER = 4.0   # 4x slower
MULTIPLIER_CRITICAL = float('inf')  # Paused


@dataclass
class ThermalState:
    """Current thermal state of the Pi."""
    temperature: float          # CPU temp in Celsius
    zone: str                   # 'cool', 'warm', 'hot', 'danger', 'critical'
    tick_multiplier: float      # How much to slow down ticks
    fan_speed: Optional[int]    # Fan speed if controllable (0-255)
    throttled: bool             # Is the CPU being throttled by the OS?
    
    def to_dict(self) -> dict:
        return {
            "temperature": self.temperature,
            "zone": self.zone,
            "tick_multiplier": self.tick_multiplier,
            "fan_speed": self.fan_speed,
            "throttled": self.throttled
        }


def read_cpu_temp() -> float:
    """
    Read CPU temperature from the Pi.
    
    Returns:
        Temperature in Celsius, or -1 if unavailable
    """
    # Primary method: thermal zone (Raspberry Pi)
    thermal_path = "/sys/class/thermal/thermal_zone0/temp"
    if os.path.exists(thermal_path):
        try:
            with open(thermal_path, 'r') as f:
                # Value is in millidegrees
                return int(f.read().strip()) / 1000.0
        except Exception as e:
            logger.warning(f"Failed to read thermal_zone0: {e}")
    
    # Fallback: vcgencmd (Raspberry Pi specific)
    try:
        import subprocess
        result = subprocess.run(
            ['vcgencmd', 'measure_temp'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            # Output: "temp=45.0'C"
            temp_str = result.stdout.strip()
            temp = float(temp_str.split('=')[1].replace("'C", ""))
            return temp
    except Exception as e:
        logger.debug(f"vcgencmd not available: {e}")
    
    # Not on a Pi or can't read temp
    return -1.0


def read_throttle_state() -> bool:
    """
    Check if the CPU is being throttled.
    
    Returns:
        True if throttled, False otherwise
    """
    try:
        import subprocess
        result = subprocess.run(
            ['vcgencmd', 'get_throttled'],
            capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            # Output: "throttled=0x0" (0 = not throttled)
            hex_val = result.stdout.strip().split('=')[1]
            return int(hex_val, 16) != 0
    except Exception:
        pass
    return False


def get_zone(temperature: float) -> Tuple[str, float]:
    """
    Determine thermal zone and tick multiplier from temperature.
    
    Args:
        temperature: CPU temperature in Celsius
        
    Returns:
        (zone_name, tick_multiplier)
    """
    if temperature < 0:
        # Can't read temp - assume warm
        return ('unknown', MULTIPLIER_WARM)
    elif temperature < TEMP_COOL:
        return ('cool', MULTIPLIER_COOL)
    elif temperature < TEMP_WARM:
        return ('warm', MULTIPLIER_WARM)
    elif temperature < TEMP_HOT:
        return ('hot', MULTIPLIER_HOT)
    elif temperature < TEMP_CRITICAL:
        return ('danger', MULTIPLIER_DANGER)
    else:
        return ('critical', MULTIPLIER_CRITICAL)


def get_thermal_state() -> ThermalState:
    """
    Get complete thermal state of the Pi.
    
    Returns:
        ThermalState with current readings
    """
    temp = read_cpu_temp()
    zone, multiplier = get_zone(temp)
    throttled = read_throttle_state()
    
    # Fan speed would come from GPIO/HAT - None for now
    fan_speed = None
    
    return ThermalState(
        temperature=temp,
        zone=zone,
        tick_multiplier=multiplier,
        fan_speed=fan_speed,
        throttled=throttled
    )


class ThermalManager:
    """
    Manages thermal state and provides tick rate adjustments.
    
    Integrates with the Clock to enforce physical constraints.
    """
    
    def __init__(self, enable_fan_control: bool = False):
        """
        Initialize thermal manager.
        
        Args:
            enable_fan_control: If True, attempt to control Pi fan via GPIO
        """
        self.enable_fan_control = enable_fan_control
        self.history: list = []  # Recent temperature readings
        self.max_history = 60    # Keep last 60 readings
        
        # Fan control (if enabled)
        self.fan_gpio_pin = 14   # Default fan control pin
        self.fan_pwm = None
        
        if enable_fan_control:
            self._init_fan_control()
    
    def _init_fan_control(self):
        """Initialize GPIO fan control."""
        try:
            import RPi.GPIO as GPIO
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(self.fan_gpio_pin, GPIO.OUT)
            self.fan_pwm = GPIO.PWM(self.fan_gpio_pin, 25000)  # 25kHz
            self.fan_pwm.start(0)
            logger.info("Fan control initialized")
        except Exception as e:
            logger.warning(f"Fan control not available: {e}")
            self.fan_pwm = None
    
    def set_fan_speed(self, speed: int):
        """
        Set fan speed (0-100).
        
        Args:
            speed: Fan speed percentage (0 = off, 100 = full)
        """
        if self.fan_pwm:
            self.fan_pwm.ChangeDutyCycle(max(0, min(100, speed)))
    
    def update(self) -> ThermalState:
        """
        Update thermal state and adjust fan if needed.
        
        Returns:
            Current ThermalState
        """
        state = get_thermal_state()
        
        # Record history
        self.history.append(state.temperature)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        # Auto fan control based on temperature
        if self.fan_pwm:
            if state.zone == 'cool':
                self.set_fan_speed(0)
            elif state.zone == 'warm':
                self.set_fan_speed(30)
            elif state.zone == 'hot':
                self.set_fan_speed(60)
            else:  # danger or critical
                self.set_fan_speed(100)
        
        return state
    
    def get_tick_multiplier(self) -> float:
        """
        Get current tick rate multiplier based on thermal state.
        
        Returns:
            Multiplier to apply to base tick interval
        """
        state = self.update()
        return state.tick_multiplier
    
    def should_pause(self) -> bool:
        """
        Check if system should pause due to critical temperature.
        
        Returns:
            True if too hot to continue
        """
        state = get_thermal_state()
        return state.zone == 'critical'
    
    def get_average_temp(self) -> float:
        """Get average temperature over history."""
        if not self.history:
            return -1.0
        return sum(self.history) / len(self.history)
    
    def cleanup(self):
        """Clean up GPIO resources."""
        if self.fan_pwm:
            self.fan_pwm.stop()
            try:
                import RPi.GPIO as GPIO
                GPIO.cleanup(self.fan_gpio_pin)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════════════════════════
# SIMULATION MODE (for testing without a Pi)
# ═══════════════════════════════════════════════════════════════════════════════

class SimulatedThermalManager(ThermalManager):
    """
    Simulated thermal manager for testing without hardware.
    
    Temperature varies based on simulated load.
    """
    
    def __init__(self):
        super().__init__(enable_fan_control=False)
        self._simulated_temp = 45.0
        self._load = 0.0  # 0.0 to 1.0
    
    def set_load(self, load: float):
        """Set simulated load (0.0 to 1.0)."""
        self._load = max(0.0, min(1.0, load))
    
    def update(self) -> ThermalState:
        """Update simulated temperature based on load."""
        # Temperature trends toward load-based target
        target = 40.0 + (self._load * 45.0)  # 40-85°C range
        
        # Gradual change
        diff = target - self._simulated_temp
        self._simulated_temp += diff * 0.1
        
        # Add some noise
        import random
        self._simulated_temp += random.uniform(-0.5, 0.5)
        self._simulated_temp = max(30.0, min(90.0, self._simulated_temp))
        
        # Record
        self.history.append(self._simulated_temp)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        zone, multiplier = get_zone(self._simulated_temp)
        
        return ThermalState(
            temperature=self._simulated_temp,
            zone=zone,
            tick_multiplier=multiplier,
            fan_speed=None,
            throttled=self._simulated_temp > TEMP_HOT
        )


def create_thermal_manager(simulated: bool = False) -> ThermalManager:
    """
    Create appropriate thermal manager.
    
    Args:
        simulated: If True, use simulated thermal manager
        
    Returns:
        ThermalManager instance
    """
    if simulated:
        return SimulatedThermalManager()
    
    # Check if we're on a Pi
    if read_cpu_temp() < 0:
        logger.info("Not on Raspberry Pi - using simulated thermal manager")
        return SimulatedThermalManager()
    
    return ThermalManager(enable_fan_control=True)
