"""
Conversation Driver - PBAI learns language through Haiku dialogue

HEAT ECONOMY:
    - Successful response: K * 1.0 (kept dialogue alive)
    - Long response: bonus heat (engaged the model)
    - Error/failure: 0 heat
    - API cost = COST_ACTION (paid to speak)

LEARNING:
    - Concepts from responses → nodes in manifold
    - Dialogue patterns → axis connections
    - Successful topics → heat accumulation
    - Language emerges from interaction patterns
"""

import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from time import time

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drivers.environment import Driver, Port, NullPort, Perception, Action, ActionResult

logger = logging.getLogger(__name__)


@dataclass
class ConversationState:
    """Current conversation state."""
    history: List[Dict[str, str]]  # [{"role": "user/assistant", "content": "..."}]
    turn_count: int = 0
    total_tokens: int = 0
    topics: List[str] = None
    
    def __post_init__(self):
        if self.topics is None:
            self.topics = []


class ConversationDriver(Driver):
    """
    Driver for conversational learning via Claude Haiku API.
    
    PBAI learns language by:
    1. Generating utterances (from manifold state)
    2. Receiving responses (from Haiku)
    3. Integrating concepts (responses → nodes)
    4. Learning patterns (successful dialogues → heat)
    
    HEAT ECONOMY:
        - Each exchange costs COST_ACTION
        - Successful response returns K * response_quality
        - Failed API call returns 0
    """
    
    DRIVER_ID = "conversation"
    DRIVER_NAME = "Haiku Conversation Driver"
    DRIVER_VERSION = "1.0.0"
    SUPPORTED_ACTIONS = ["speak", "listen", "reset", "reflect"]
    HEAT_SCALE = 1.0
    
    def __init__(self, port: Port = None, config: Dict[str, Any] = None,
                 api_key: str = None, manifold=None):
        """
        Initialize conversation driver.
        
        Args:
            port: Port (not used for API)
            config: Configuration dict
            api_key: Anthropic API key (or from env ANTHROPIC_API_KEY)
            manifold: PBAI manifold for learning (creates DriverNode)
        """
        super().__init__(port or NullPort("conversation"), config or {}, manifold=manifold)
        
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.model = config.get("model", "claude-3-haiku-20240307") if config else "claude-3-haiku-20240307"
        self.max_tokens = config.get("max_tokens", 256) if config else 256
        
        # Conversation state
        self.state = ConversationState(history=[])
        
        # System prompt for Haiku
        self.system_prompt = """You are having a conversation with PBAI, an experimental AI system that is learning language through dialogue. Keep responses concise but natural. PBAI may say unusual things as it learns - respond helpfully and guide the conversation. Extract and mention key concepts when relevant."""
        
        # Track concepts seen
        self._seen_concepts = set()
    
    def initialize(self) -> bool:
        """Initialize the driver."""
        if not self.api_key:
            logger.error("No API key - set ANTHROPIC_API_KEY environment variable")
            return False
        
        if self.port:
            self.port.connect()
        
        self.active = True
        self.state = ConversationState(history=[])
        logger.info("ConversationDriver initialized")
        return True
    
    def shutdown(self) -> bool:
        """Shutdown the driver."""
        if self.port:
            self.port.disconnect()
        self.active = False
        return True
    
    def perceive(self) -> Perception:
        """
        Get current perception (last response from Haiku).
        Also feeds to DriverNode for learning.
        """
        from core.node_constants import K
        
        # Get last assistant message
        last_response = ""
        for msg in reversed(self.state.history):
            if msg["role"] == "assistant":
                last_response = msg["content"]
                break
        
        # Extract concepts from response
        concepts = self._extract_concepts(last_response)
        
        # Novelty heat for new concepts
        new_concepts = [c for c in concepts if c not in self._seen_concepts]
        novelty_heat = len(new_concepts) * 0.1 * K
        self._seen_concepts.update(new_concepts)
        
        perception = Perception(
            entities=concepts,
            locations=["conversation"],
            properties={
                "last_response": last_response,
                "turn_count": self.state.turn_count,
                "history_length": len(self.state.history),
                "topics": self.state.topics[-5:] if self.state.topics else []
            },
            events=["new_response"] if last_response else [],
            heat_value=novelty_heat,
            source_driver=self.DRIVER_ID
        )
        
        # Feed to DriverNode for learning
        self.feed_perception(perception)
        
        return perception
    
    def act(self, action: Action) -> ActionResult:
        """
        Execute conversation action.
        
        Actions:
            - speak: Send message to Haiku
            - listen: Just get perception (no API call)
            - reset: Clear conversation history
            - reflect: Ask Haiku to summarize conversation
        
        Also feeds results to DriverNode for learning.
        """
        from core.node_constants import K
        
        action_type = action.action_type
        
        if action_type == "reset":
            self.state = ConversationState(history=[])
            return ActionResult(
                success=True,
                outcome="Conversation reset",
                heat_value=0.0
            )
        
        elif action_type == "listen":
            # Just return current state, no API call
            return ActionResult(
                success=True,
                outcome="Listening",
                heat_value=0.0
            )
        
        elif action_type == "speak":
            message = action.parameters.get("message", "")
            if not message:
                # Generate message from target or default
                message = action.target or "Hello"
            
            result = self._send_message(message)
            # Feed to DriverNode for learning
            self.feed_result(result, action)
            return result
        
        elif action_type == "reflect":
            # Ask Haiku to reflect on conversation
            reflect_prompt = "Please briefly summarize our conversation so far and note any key concepts or patterns."
            result = self._send_message(reflect_prompt)
            # Feed to DriverNode for learning
            self.feed_result(result, action)
            return result
        
        return ActionResult(
            success=False,
            outcome=f"Unknown action: {action_type}",
            heat_value=0.0
        )
    
    def _send_message(self, message: str) -> ActionResult:
        """
        Send message to Haiku API.
        
        Returns:
            ActionResult with heat based on response quality
        """
        from core.node_constants import K
        
        # Add user message to history
        self.state.history.append({"role": "user", "content": message})
        self.state.turn_count += 1
        
        try:
            response = self._call_api(message)
            
            if response:
                # Add assistant response to history
                self.state.history.append({"role": "assistant", "content": response})
                
                # Extract concepts for topic tracking
                concepts = self._extract_concepts(response)
                self.state.topics.extend(concepts[:3])
                
                # Calculate heat based on response quality
                # Longer, more engaged responses = more heat
                word_count = len(response.split())
                base_heat = K * 1.0
                length_bonus = min(K * 0.5, word_count * 0.01 * K)
                
                heat_value = self.scale_heat(base_heat + length_bonus)
                
                return ActionResult(
                    success=True,
                    outcome=f"Response: {response[:100]}..." if len(response) > 100 else f"Response: {response}",
                    heat_value=heat_value,
                    changes={
                        "response": response,
                        "concepts": concepts,
                        "word_count": word_count
                    }
                )
            else:
                return ActionResult(
                    success=False,
                    outcome="Empty response from API",
                    heat_value=0.0
                )
                
        except Exception as e:
            logger.error(f"API call failed: {e}")
            return ActionResult(
                success=False,
                outcome=f"API error: {str(e)}",
                heat_value=0.0
            )
    
    def _call_api(self, message: str) -> Optional[str]:
        """
        Call Anthropic API.
        
        Returns:
            Response text or None on failure
        """
        import urllib.request
        import json
        
        url = "https://api.anthropic.com/v1/messages"
        
        # Build messages (include history for context)
        messages = self.state.history[-10:]  # Last 10 messages for context
        
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": self.system_prompt,
            "messages": messages
        }
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(url, data=data, headers=headers)
            
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode())
                
                if "content" in result and len(result["content"]) > 0:
                    return result["content"][0]["text"]
                    
        except Exception as e:
            logger.error(f"API request failed: {e}")
            raise
        
        return None
    
    def _extract_concepts(self, text: str) -> List[str]:
        """
        Extract key concepts from text.
        
        Simple extraction: nouns and key phrases.
        Could be enhanced with NLP.
        """
        if not text:
            return []
        
        # Simple word extraction (could use NLP for better results)
        words = text.lower().split()
        
        # Filter: keep longer words, remove common ones
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'must', 'shall',
            'can', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
            'from', 'up', 'about', 'into', 'through', 'during', 'before',
            'after', 'above', 'below', 'between', 'under', 'again',
            'further', 'then', 'once', 'here', 'there', 'when', 'where',
            'why', 'how', 'all', 'each', 'few', 'more', 'most', 'other',
            'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same',
            'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if',
            'or', 'because', 'as', 'until', 'while', 'this', 'that',
            'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they',
            'what', 'which', 'who', 'whom', 'your', 'his', 'her', 'its',
            'our', 'their', 'my', "i'm", "you're", "it's", "that's"
        }
        
        # Clean and filter
        concepts = []
        for word in words:
            # Remove punctuation
            clean = ''.join(c for c in word if c.isalnum())
            if len(clean) >= 4 and clean not in stopwords:
                concepts.append(clean)
        
        # Dedupe while preserving order
        seen = set()
        unique = []
        for c in concepts:
            if c not in seen:
                seen.add(c)
                unique.append(c)
        
        return unique[:10]  # Max 10 concepts
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONVENIENCE METHODS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def say(self, message: str) -> ActionResult:
        """Convenience: send a message."""
        return self.act(Action(action_type="speak", parameters={"message": message}))
    
    def reset_conversation(self) -> ActionResult:
        """Convenience: reset conversation."""
        return self.act(Action(action_type="reset"))
    
    def get_history(self) -> List[Dict[str, str]]:
        """Get conversation history."""
        return self.state.history.copy()


def create_conversation_driver(api_key: str = None) -> ConversationDriver:
    """
    Create a conversation driver.
    
    Args:
        api_key: Anthropic API key (or uses env var)
        
    Returns:
        Configured ConversationDriver
    """
    driver = ConversationDriver(api_key=api_key)
    return driver
