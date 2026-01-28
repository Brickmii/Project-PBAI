"""
PBAI Driver Template - Proper DriverNode Architecture

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │                  Python Driver (this file)                  │
    │  - Thin adapter for I/O with external system               │
    │  - Translates raw data ↔ VisionReport/MotorAction          │
    │  - Does NOT store knowledge - just passes it through       │
    └───────────────────────────┬─────────────────────────────────┘
                                │ uses
                                ▼
    ┌─────────────────────────────────────────────────────────────┐
    │                 DriverNode (in core/)                       │
    │  - Lives in the manifold                                    │
    │  - Persists to drivers/{name}/ folder                       │
    │  - Stores: states, motors, plans (learned knowledge)        │
    │  - Connected to other concepts via axes                     │
    │  - Enables transfer: "maze" knowledge → "roadmap" driver    │
    └─────────────────────────────────────────────────────────────┘

WHY THIS MATTERS:
    When PBAI encounters a new game that's similar to one it knows:
    1. SearchNode finds similar DriverNodes (e.g., "maze" ≈ "roadmap")
    2. CapabilityNode detects shared capabilities
    3. New DriverNode can link to existing knowledge
    4. Learning transfers automatically

CREATING A NEW DRIVER:
    1. Copy this file to: {name}_driver.py
    2. Implement YourPort (raw I/O)
    3. Implement YourDriver (perception encoding, action decoding)
    4. The DriverNode is created automatically when first used

FOLDER STRUCTURE AFTER USE:
    drivers/
    ├── your_driver.py          # This Python code (thin adapter)
    └── your/                   # Created by DriverNode
        ├── frame.json          # Core node data
        ├── states/             # Learned perception patterns
        │   └── state_key.json
        ├── motors/             # Available actions
        │   └── action_name.json
        └── plans/              # Learned sequences
            └── plan_name.json
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .environment import Driver, Port, Perception, Action, ActionResult, PortMessage, PortState

# Import DriverNode from core - this is where knowledge lives
from core.driver_node import (
    DriverNode, 
    VisionReport, 
    MotorAction, 
    MotorType,
    press, hold_key, release_key, look, mouse_hold, mouse_release, wait
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# PORT - Raw I/O with external system
# ═══════════════════════════════════════════════════════════════════════════════

class TemplatePort(Port):
    """
    Handles raw communication with [YOUR ENVIRONMENT].
    
    The Port is ONLY for I/O - no game logic, no learning.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__("template_port", config)
        # Your connection state
        # self.socket = None
        # self.process = None
    
    def connect(self) -> bool:
        """Establish connection."""
        try:
            # TODO: Your connection logic
            self.state = PortState.CONNECTED
            return True
        except Exception as e:
            logger.error(f"Connection failed: {e}")
            self.state = PortState.ERROR
            return False
    
    def disconnect(self) -> bool:
        """Close connection."""
        try:
            # TODO: Your disconnection logic
            self.state = PortState.DISCONNECTED
            return True
        except Exception:
            return False
    
    def send(self, message: PortMessage) -> bool:
        """Send to external system."""
        if self.state != PortState.CONNECTED:
            return False
        # TODO: Your send logic
        return True
    
    def receive(self, timeout: float = 1.0) -> Optional[PortMessage]:
        """Receive from external system."""
        if self.state != PortState.CONNECTED:
            return None
        # TODO: Your receive logic
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# DRIVER - Thin adapter that uses DriverNode for knowledge
# ═══════════════════════════════════════════════════════════════════════════════

class TemplateDriver(Driver):
    """
    Driver for [YOUR ENVIRONMENT].
    
    This class is a THIN ADAPTER:
    - Encodes raw observations → VisionReport
    - Decodes Action → raw commands
    - Delegates all learning/knowledge to DriverNode
    
    The DriverNode (in core/) handles:
    - State storage and matching
    - Motor pattern registration
    - Plan learning and execution
    - Persistence to drivers/{name}/ folder
    """
    
    # Driver metadata
    DRIVER_ID = "template"
    DRIVER_NAME = "Template Driver"
    DRIVER_VERSION = "1.0.0"
    
    # Actions this environment supports
    SUPPORTED_ACTIONS = ["move_up", "move_down", "move_left", "move_right", "interact"]
    
    def __init__(self, manifold=None, config: Dict[str, Any] = None):
        """
        Initialize driver.
        
        Args:
            manifold: The ONE PBAI manifold (required for DriverNode)
            config: Environment-specific configuration
        """
        port = TemplatePort(config)
        super().__init__(port, manifold=manifold)
        
        # Create DriverNode - this is where knowledge lives
        # If drivers/template/ exists, it loads. Otherwise creates.
        self.driver_node: Optional[DriverNode] = None
        if manifold:
            self.driver_node = DriverNode("template", manifold)
            self._register_motors()
    
    def _register_motors(self):
        """
        Register available motor patterns with DriverNode.
        
        Motors are atomic actions PBAI can perform.
        Call this once during initialization.
        """
        if not self.driver_node:
            return
        
        # Register motors for this environment
        # These persist to drivers/template/motors/
        
        # Example: Movement motors
        self.driver_node.register_motor("move_up", press("w"))
        self.driver_node.register_motor("move_down", press("s"))
        self.driver_node.register_motor("move_left", press("a"))
        self.driver_node.register_motor("move_right", press("d"))
        
        # Example: Interaction motors
        self.driver_node.register_motor("interact", press("e"))
        self.driver_node.register_motor("use", mouse_hold("left"))
        
        logger.info(f"Registered {len(self.driver_node.motor_patterns)} motors")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ENCODING: Raw observations → VisionReport → DriverNode
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _encode_observation(self, raw_data: Any) -> VisionReport:
        """
        Convert raw observation to VisionReport.
        
        VisionReport is what "Sonnet sees" - structured perception
        that DriverNode can learn from.
        
        Args:
            raw_data: Whatever your environment provides
            
        Returns:
            VisionReport for DriverNode
        """
        # TODO: Extract meaningful features from raw_data
        
        # Example structure:
        return VisionReport(
            timestamp=0.0,
            description="",  # Natural language description
            objects=[],      # [{"type": "enemy", "direction": "north", "distance": 5}]
            inventory={},    # {"wood": 3, "stone": 0}
            status={},       # {"health": 8, "hunger": 10}
            threats=[]       # [{"type": "zombie", "direction": "behind"}]
        )
    
    def perceive(self) -> Perception:
        """
        Get current perception from environment.
        
        Flow:
            Raw data → VisionReport → DriverNode.see() → Perception
        """
        # Get raw data from port
        message = self.port.receive()
        if not message:
            return Perception(source_driver=self.DRIVER_ID)
        
        # Encode to VisionReport
        vision = self._encode_observation(message.payload)
        
        # Feed to DriverNode (learns state patterns)
        state_key = None
        if self.driver_node:
            state_key = self.driver_node.see(vision)
        
        # Build Perception for EnvironmentCore
        return Perception(
            entities=[state_key] if state_key else [],
            properties={
                "state_key": state_key or "unknown",
                "description": vision.description,
                # Add environment-specific features for context
                # These enable generalization via DecisionNode
            },
            source_driver=self.DRIVER_ID
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DECODING: Action → MotorAction → Raw commands
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _decode_action(self, action: Action) -> Optional[MotorAction]:
        """
        Convert Action to MotorAction.
        
        Args:
            action: High-level action from DecisionNode
            
        Returns:
            MotorAction that DriverNode can execute
        """
        if not self.driver_node:
            return None
        
        # Look up motor pattern by name
        action_name = action.target or action.action_type
        return self.driver_node.motor_patterns.get(action_name)
    
    def act(self, action: Action) -> ActionResult:
        """
        Execute action in environment.
        
        Flow:
            Action → MotorAction → DriverNode.act() → Raw command → Port
        """
        action_name = action.target or action.action_type
        
        # Get motor pattern
        motor = self._decode_action(action)
        if not motor:
            return ActionResult(
                success=False,
                heat_value=-0.1,
                message=f"Unknown action: {action_name}"
            )
        
        # Execute via DriverNode (tracks history for learning)
        success = False
        if self.driver_node:
            success = self.driver_node.act(motor)
        
        # TODO: Observe result from environment
        # result_message = self.port.receive()
        
        return ActionResult(
            success=success,
            heat_value=0.1 if success else -0.1,
            message=f"Executed {action_name}"
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LIFECYCLE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def initialize(self) -> bool:
        """Initialize driver and connect to environment."""
        if not self.port.connect():
            return False
        
        self.active = True
        
        # Save DriverNode state
        if self.driver_node:
            self.driver_node.save()
        
        return True
    
    def shutdown(self) -> bool:
        """Shutdown driver."""
        self.active = False
        
        # Save learned knowledge
        if self.driver_node:
            self.driver_node.save()
        
        self.port.disconnect()
        return True
    
    def get_supported_actions(self) -> List[str]:
        """Return available actions."""
        if self.driver_node:
            return list(self.driver_node.motor_patterns.keys())
        return self.SUPPORTED_ACTIONS


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_template_driver(manifold, **config) -> TemplateDriver:
    """
    Create a Template driver.
    
    Args:
        manifold: PBAI manifold (required)
        **config: Environment configuration
        
    Returns:
        Configured TemplateDriver
    """
    return TemplateDriver(manifold=manifold, config=config)
