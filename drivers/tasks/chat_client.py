#!/usr/bin/env python3
"""
PBAI Thermal Manifold - Chat Client (v2)

A GUI chat interface for PBAI with full conception architecture:
- Chat panel (input + response history)
- Node info panel (all nodes with conceptions)
- Conception graph panel (semantic relationships)
- Capability upgrade detection (R→O→M→G)

Features:
- Relationship extraction from natural language
- Question answering via conception traversal
- Persistent memory (saves/loads growth maps)
- Automatic capability upgrades

Requirements:
    pip install pygame-ce

Usage:
    python -m drivers.tasks.chat_client
    python -m drivers.tasks.chat_client --no-llm
    python -m drivers.tasks.chat_client --fresh   # Start with empty manifold
"""

import sys
import os
import math
import logging
import re
from time import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core import (
    Manifold, search, K,
    Node, SelfNode, Axis, Frame,
    get_axis_coordinates, CapabilityManager, get_capability_manager,
    CAPABILITY_RIGHTEOUS, CAPABILITY_PROPER, CAPABILITY_ORDERED, CAPABILITY_MOVABLE, CAPABILITY_GRAPHIC
)


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_concepts(text: str) -> list:
    """Extract concepts from text (simple word extraction)."""
    import re
    # Remove punctuation, split on whitespace
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    # Filter stopwords
    stopwords = {'the', 'and', 'but', 'for', 'are', 'was', 'were', 'been', 'being',
                 'have', 'has', 'had', 'does', 'did', 'will', 'would', 'could', 'should',
                 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used',
                 'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose',
                 'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
                 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'than',
                 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once',
                 'with', 'from', 'into', 'about', 'your', 'you', 'they', 'them', 'their'}
    return [w for w in words if w not in stopwords]


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

WINDOW_WIDTH = 1400
WINDOW_HEIGHT = 900

# Panel dimensions
CHAT_PANEL_WIDTH = 500
CONCEPTION_PANEL_WIDTH = 450
NODE_PANEL_WIDTH = 450

# Colors
COLOR_BG = (30, 30, 35)
COLOR_PANEL_BG = (40, 40, 48)
COLOR_PANEL_BORDER = (60, 60, 70)
COLOR_TEXT = (220, 220, 220)
COLOR_TEXT_DIM = (140, 140, 150)
COLOR_INPUT_BG = (50, 50, 60)
COLOR_USER_MSG = (100, 180, 255)
COLOR_PBAI_MSG = (100, 255, 150)
COLOR_SYSTEM_MSG = (255, 200, 100)
COLOR_NODE_HOT = (255, 100, 100)
COLOR_NODE_WARM = (255, 200, 100)
COLOR_NODE_COOL = (100, 200, 255)
COLOR_NODE_SELF = (255, 255, 100)
COLOR_CAP_R = (180, 180, 180)  # Righteous - gray
COLOR_CAP_O = (100, 200, 255)  # Ordered - blue
COLOR_CAP_M = (255, 200, 100)  # Movable - yellow
COLOR_CAP_G = (100, 255, 150)  # Graphic - green
COLOR_TOGGLE_ON = (100, 255, 150)
COLOR_TOGGLE_OFF = (255, 100, 100)

# Use unified growth path from constants - ONE brain for all tasks
from core.node_constants import get_growth_path
GROWTH_PATH = get_growth_path("growth_map.json")


@dataclass
class ChatMessage:
    """A message in the chat history."""
    sender: str  # "user", "pbai", "system"
    text: str
    timestamp: float


# ═══════════════════════════════════════════════════════════════════════════════
# PBAI CHAT ENGINE (with Conception Architecture)
# ═══════════════════════════════════════════════════════════════════════════════

class PBAIChatEngine:
    """
    PBAI Chat Engine with full conception architecture.
    
    Processes user input to:
    1. Extract relationships ("X is Y", "Your name is Z")
    2. Build conceptions (semantic links with capability hierarchy)
    3. Answer questions by traversing conceptions
    4. Automatically detect and apply capability upgrades
    """
    
    def __init__(self, use_llm: bool = False, growth_path: str = GROWTH_PATH,
                 auto_persist: bool = True, fresh: bool = False):
        """
        Initialize the chat engine.
        
        Args:
            use_llm: Whether to use LLM for formatting responses
            growth_path: Path to save/load growth maps
            auto_persist: Whether to auto-save after each input
            fresh: If True, delete existing and start fresh (for testing only)
        """
        from core import get_pbai_manifold, reset_pbai_manifold
        
        self.growth_path = growth_path
        self.auto_persist = auto_persist
        self.use_llm = use_llm
        
        if fresh:
            # Fresh start requested - delete existing and reset singleton
            if os.path.exists(growth_path):
                os.remove(growth_path)
                logger.info("Deleted existing growth map (fresh start)")
            reset_pbai_manifold()  # Reset singleton so get_pbai_manifold will birth
        
        # Get the ONE PBAI manifold (loads existing or births on first run)
        self.manifold = get_pbai_manifold(growth_path)
        logger.info(f"PBAI mind ready: {len(self.manifold.nodes)} nodes")
        
        # Capability detector for automatic upgrades
        self.capability_detector = CapabilityManager(self.manifold)
        
        # Conversation history
        self.history: List[Tuple[str, str]] = []
        
        # LLM setup (optional)
        self.llm = None
        if use_llm:
            self._init_llm()
    
    def _try_load(self) -> bool:
        """Legacy method - kept for compatibility but not used."""
        return os.path.exists(self.growth_path)
    
    def save(self) -> str:
        """Save current growth map."""
        os.makedirs(os.path.dirname(self.growth_path) or ".", exist_ok=True)
        self.manifold.save_growth_map(self.growth_path)
        return f"Saved to {self.growth_path}"
    
    def reset(self) -> str:
        """Reset manifold to fresh state (for testing only)."""
        from core import get_pbai_manifold, reset_pbai_manifold
        
        if os.path.exists(self.growth_path):
            os.remove(self.growth_path)
        reset_pbai_manifold()  # Reset singleton
        self.manifold = get_pbai_manifold(self.growth_path)  # Will birth fresh
        self.capability_detector = CapabilityManager(self.manifold)
        self.history = []
        return "Reset complete. Fresh start."
    
    def _init_llm(self):
        """Initialize LLM for response formatting."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_name = "Qwen/Qwen2.5-3B-Instruct"
            logger.info(f"Loading LLM: {model_name}...")
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                torch_dtype=torch.float16
            )
            
            def call_llm(prompt: str) -> str:
                messages = [{"role": "user", "content": prompt}]
                text = self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                return self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:],
                    skip_special_tokens=True
                )
            
            self.llm = call_llm
            logger.info("LLM ready")
            
        except Exception as e:
            logger.warning(f"LLM not available: {e}")
            self.use_llm = False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INPUT PROCESSING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate response.
        
        Flow:
        1. Extract relationships ("X is Y")
        2. Build conceptions from relationships
        3. Process concepts into manifold
        4. Try to answer if it's a question
        5. Check for capability upgrades
        6. Generate response
        7. Process response back into manifold (PBAI learns from its own outputs)
        """
        concepts = extract_concepts(user_input)
        
        # 1. Extract and build relationships from user input
        relationships = self._extract_relationships(user_input)
        for subject, predicate, obj, direction in relationships:
            self._build_relationship(subject, predicate, obj, direction)
        
        # 2. Process concepts into manifold
        for concept in concepts:
            search(concept, self.manifold)
        
        # 3. Try to answer question from conceptions
        answer = self._try_answer_question(user_input, concepts)
        if answer:
            # Still process the answer for learning
            self._process_own_response(answer)
            self.history.append((user_input, answer))
            if self.auto_persist:
                self.save()
            return answer
        
        # 4. Check for capability upgrades (now handled by CapabilityManager)
        upgrades = []
        
        # 5. Generate response
        if self.use_llm and self.llm:
            response = self._format_with_llm(user_input, concepts, relationships, upgrades)
        else:
            response = self._format_direct(user_input, concepts, relationships, upgrades)
        
        # 6. Process response back into manifold (PBAI internalizes its own statements)
        self._process_own_response(response)
        
        self.history.append((user_input, response))
        
        if self.auto_persist:
            self.save()
        
        return response
    
    def _process_own_response(self, response: str):
        """
        Process PBAI's own response back into the manifold.
        
        This is crucial - PBAI must learn from its own outputs, not just user inputs.
        When PBAI says "my favorite color is green", that should become a conception.
        """
        # Convert "my X" to "self X" and "I am" to "self is" for self-reference
        # This makes PBAI's first-person statements become self-conceptions
        processed = response.lower()
        processed = re.sub(r'\bmy\b', 'your', processed)  # "my name" -> "your name"
        processed = re.sub(r'\bi am\b', 'you are', processed)  # "I am" -> "you are"
        processed = re.sub(r'\bi\b', 'you', processed)  # "I like" -> "you like"
        
        # Extract relationships from PBAI's own statements
        relationships = self._extract_relationships(processed)
        for subject, predicate, obj, direction in relationships:
            self._build_relationship(subject, predicate, obj, direction)
            logger.debug(f"Self-learned: {subject} --{predicate}--> {obj}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RELATIONSHIP EXTRACTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_relationships(self, text: str) -> List[Tuple[str, str, str, str]]:
        """
        Extract relationships from natural language text.
        
        Returns list of (subject, predicate, object, direction) tuples.
        
        Direction semantics:
        - 'n' = positive assertion (is, has, equals)
        - 's' = negative assertion (is not, doesn't)
        - 'u' = generalization (is a type of)
        - 'd' = specification (is an example of)
        """
        relationships = []
        lower = text.lower().strip()
        
        # Skip if this is a question
        is_question = (lower.endswith('?') or 
                      lower.startswith(('what', 'who', 'where', 'when', 'why', 'how', 
                                       'do ', 'does ', 'is ', 'are ', 'can ')))
        
        if is_question:
            return relationships
        
        # Pattern: "Your X is Y" (property of PBAI/self)
        # Handles multi-word properties like "favorite color"
        for match in re.finditer(r"(?:^|[.!]\s*)your\s+([\w\s]+?)\s+is\s+(\w+)", lower):
            prop, value = match.groups()
            # Use underscore for multi-word predicates
            prop = prop.strip().replace(' ', '_')
            relationships.append(("self", prop, value, 'n'))
        
        # Pattern: "My X is Y" (property of user)
        # Handles multi-word properties like "favorite color"
        for match in re.finditer(r"(?:^|[.!]\s*)my\s+([\w\s]+?)\s+is\s+(\w+)", lower):
            prop, value = match.groups()
            prop = prop.strip().replace(' ', '_')
            relationships.append(("user", prop, value, 'n'))
        
        # Pattern: "You are X" (identity of PBAI)
        for match in re.finditer(r"(?:^|[.!]\s*)you\s+are\s+(\w+)", lower):
            value = match.group(1)
            skip = ('a', 'an', 'the', 'not', 'being', 'going', 'doing', 
                   'welcome', 'right', 'wrong', 'here', 'there')
            if value not in skip and len(value) > 1:
                relationships.append(("self", "identity", value, 'n'))
                relationships.append(("self", "name", value, 'n'))
        
        # Pattern: "I am X" (identity of user)
        for match in re.finditer(r"(?:^|[.!]\s*)i\s+am\s+(\w+)", lower):
            value = match.group(1)
            skip = ('a', 'an', 'the', 'not', 'going', 'doing', 'sure')
            if value not in skip and len(value) > 1:
                relationships.append(("user", "identity", value, 'n'))
                relationships.append(("user", "name", value, 'n'))
        
        # Pattern: "X is your Y" (X has role Y for self)
        # Handles multi-word roles like "favorite color"
        for match in re.finditer(r"(\w+)\s+is\s+your\s+([\w\s]+?)(?:\.|,|!|$)", lower):
            value, role = match.groups()
            skip = ('it', 'this', 'that', 'what', 'who', 'which')
            if value not in skip:
                role = role.strip().replace(' ', '_')
                relationships.append(("self", role, value, 'n'))
        
        # Pattern: "I created you" (creator relationship)
        if re.search(r"i\s+created\s+you", lower):
            relationships.append(("self", "creator", "user", 'n'))
            relationships.append(("user", "created", "self", 'n'))
        
        # Pattern: "X created Y"
        for match in re.finditer(r"(\w+)\s+created\s+(\w+)", lower):
            creator, created = match.groups()
            if creator != "i" and created != "you":
                relationships.append((created, "creator", creator, 'n'))
        
        # Pattern: "X is a/an Y" (type relationship)
        for match in re.finditer(r"(\w+)\s+is\s+(?:a|an)\s+(\w+)", lower):
            instance, category = match.groups()
            skip = ('it', 'this', 'that', 'what', 'who', 'which', 'there')
            if instance not in skip:
                relationships.append((instance, "type", category, 'u'))
        
        # Pattern: "X is not Y" (negative assertion)
        for match in re.finditer(r"(\w+)\s+(?:is\s+not|isn't|am\s+not)\s+(\w+)", lower):
            subj, obj = match.groups()
            if subj == 'i':
                subj = 'user'
            if obj not in ('a', 'an', 'the'):
                relationships.append((subj, "not", obj, 's'))
        
        # Pattern: "I like/love X" (user preference)
        for match in re.finditer(r"i\s+(?:like|love)\s+(\w+)", lower):
            value = match.group(1)
            skip = ('to', 'the', 'a', 'an', 'it', 'this', 'that')
            if value not in skip:
                relationships.append(("user", "likes", value, 'n'))
        
        # Pattern: "You like/love X" (self preference)
        for match in re.finditer(r"you\s+(?:like|love)\s+(\w+)", lower):
            value = match.group(1)
            skip = ('to', 'the', 'a', 'an', 'it', 'this', 'that')
            if value not in skip:
                relationships.append(("self", "likes", value, 'n'))
        
        # Pattern: "X is my favorite" (user favorite)
        for match in re.finditer(r"(\w+)\s+is\s+my\s+favorite", lower):
            value = match.group(1)
            skip = ('it', 'this', 'that', 'what')
            if value not in skip:
                relationships.append(("user", "favorite", value, 'n'))
        
        # Pattern: "X is your favorite" (self favorite)
        for match in re.finditer(r"(\w+)\s+is\s+your\s+favorite", lower):
            value = match.group(1)
            skip = ('it', 'this', 'that', 'what')
            if value not in skip:
                relationships.append(("self", "favorite", value, 'n'))
        
        return relationships
    
    def _build_relationship(self, subject: str, predicate: str, obj: str, direction: str):
        """
        Build a conception in the manifold from extracted relationship.
        
        Creates nodes if needed, creates/strengthens conceptions.
        """
        # Resolve subject node
        if subject == "self":
            subj_node = self.manifold.self_node
        else:
            subj_node = self._ensure_node(subject)
        
        # Resolve object node
        if obj == "self":
            obj_node = self.manifold.self_node
        else:
            obj_node = self._ensure_node(obj)
        
        if not subj_node or not obj_node:
            return
        
        # Map direction to polarity
        polarity = -1 if direction == 's' else +1
        
        # Create or strengthen conception
        conception = subj_node.conceive(predicate, obj_node.id, polarity)
        
        if conception.traversal_count > 1:
            logger.info(f"Strengthened: {subject} --{predicate}--> {obj} (×{conception.traversal_count})")
        else:
            logger.info(f"Conceived: {subject} --{predicate}--> {obj}")
        
        # Add heat to involved nodes
        if hasattr(subj_node, 'add_heat') and subject != "self":
            subj_node.add_heat(K * 0.5)
        obj_node.add_heat(K * 0.5)
    
    def _ensure_node(self, concept: str) -> Optional[Node]:
        """Get existing node or create new one."""
        node = self.manifold.get_node_by_concept(concept)
        if not node:
            result = search(concept, self.manifold)
            node = result.center
        return node
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QUESTION ANSWERING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _try_answer_question(self, user_input: str, concepts: List[str]) -> Optional[str]:
        """
        Try to answer a question by following conceptions.
        
        "What is your name?" → self --name--> ?
        "What is your favorite color?" → self --favorite_color--> ?
        """
        lower = user_input.lower()
        
        # "What is your X?" or "What's your X?" (multi-word properties)
        match = re.search(r"what(?:'s|\s+is)\s+your\s+([\w\s]+?)(?:\?|$)", lower)
        if match:
            prop = match.group(1).strip().replace(' ', '_')
            return self._query_property("self", prop)
        
        # "Who is your X?" (multi-word properties)
        match = re.search(r"who\s+is\s+your\s+([\w\s]+?)(?:\?|$)", lower)
        if match:
            prop = match.group(1).strip().replace(' ', '_')
            return self._query_property("self", prop)
        
        # "Who are you?" or "What are you?"
        if re.search(r"(?:who|what)\s+are\s+you", lower):
            return self._query_property("self", "identity") or self._query_property("self", "name")
        
        # "Do you have a/an X?" (multi-word properties)
        match = re.search(r"do\s+you\s+have\s+(?:an?\s+)?([\w\s]+?)(?:\?|$)", lower)
        if match:
            prop = match.group(1).strip().replace(' ', '_')
            return self._query_existence("self", prop)
        
        # "What is my X?" (multi-word properties)
        match = re.search(r"what(?:'s|\s+is)\s+my\s+([\w\s]+?)(?:\?|$)", lower)
        if match:
            prop = match.group(1).strip().replace(' ', '_')
            return self._query_property("user", prop)
        
        # "Who am I?"
        if re.search(r"who\s+am\s+i", lower):
            return self._query_property("user", "identity") or self._query_property("user", "name")
        
        # "Who created you?"
        if re.search(r"who\s+created\s+you", lower):
            return self._query_property("self", "creator")
        
        # "Do you like X?" / "What do you like?"
        match = re.search(r"do\s+you\s+like\s+(\w+)", lower)
        if match:
            return self._query_property("self", "likes")
        
        if re.search(r"what\s+do\s+you\s+like", lower):
            return self._query_property("self", "likes")
        
        return None
    
    def _query_property(self, subject: str, prop: str) -> Optional[str]:
        """Query a property by following conceptions."""
        if subject == "self":
            subj_node = self.manifold.self_node
        else:
            subj_node = self.manifold.get_node_by_concept(subject)
        
        if not subj_node:
            return None
        
        conception = subj_node.get_conception(prop)
        if conception:
            target = self.manifold.get_node(conception.target_id)
            if target:
                count = conception.traversal_count
                cap = conception.capability[0].upper()
                # Format predicate for display (favorite_color -> favorite color)
                display_prop = prop.replace('_', ' ')
                
                if subject == "self":
                    if prop in ("name", "identity"):
                        return f"My {display_prop} is {target.concept}." if count >= 2 else f"I think my {display_prop} is {target.concept}."
                    elif prop == "creator":
                        if target.concept == "user":
                            return "You created me." if count >= 2 else "I believe you created me."
                        return f"{target.concept.capitalize()} created me."
                    elif prop == "likes":
                        return f"I like {target.concept}." if count >= 2 else f"I think I like {target.concept}."
                    elif prop == "favorite":
                        return f"My favorite is {target.concept}." if count >= 2 else f"I think my favorite is {target.concept}."
                    else:
                        return f"My {display_prop} is {target.concept}." if count >= 2 else f"I think my {display_prop} is {target.concept}."
                elif subject == "user":
                    return f"Your {display_prop} is {target.concept}."
        
        # Not found - format predicate for display
        display_prop = prop.replace('_', ' ')
        if subject == "self":
            return f"I don't know my {display_prop} yet."
        return None
    
    def _query_existence(self, subject: str, prop: str) -> Optional[str]:
        """Check if subject has a property."""
        if subject == "self":
            subj_node = self.manifold.self_node
        else:
            subj_node = self.manifold.get_node_by_concept(subject)
        
        if not subj_node:
            return None
        
        conception = subj_node.get_conception(prop)
        if conception:
            target = self.manifold.get_node(conception.target_id)
            if target:
                display_prop = prop.replace('_', ' ')
                return f"Yes, my {display_prop} is {target.concept}."
        
        display_prop = prop.replace('_', ' ')
        return f"I don't know if I have a {display_prop} yet."
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RESPONSE FORMATTING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _format_direct(self, user_input: str, concepts: List[str], 
                       relationships: List[Tuple], upgrades: List) -> str:
        """Format response directly from manifold state."""
        parts = []
        
        # Report relationships learned
        if relationships:
            rel_strs = [f"{s}→{p}→{o}" for s, p, o, d in relationships[:3]]
            parts.append(f"[Learned: {', '.join(rel_strs)}]")
        
        # Report concepts processed
        if concepts:
            parts.append(f"[Concepts: {', '.join(concepts[:5])}]")
        
        # Report upgrades
        if upgrades:
            for u in upgrades:
                parts.append(f"[Upgraded: {u.source_concept}.{u.conception_predicate} → {u.new_capability[0].upper()}]")
        
        # Show what PBAI knows
        self_concs = []
        for pred, conc in self.manifold.self_node.frame.semantic_axes.items():
            target = self.manifold.get_node(conc.target_id)
            if target:
                cap = conc.capability[0].upper()
                self_concs.append(f"{pred}→{target.concept}[{cap}]")
        if self_concs:
            parts.append(f"[Self knows: {', '.join(self_concs[:5])}]")
        
        if not parts:
            parts.append("[Processed input]")
        
        return "\n".join(parts)
    
    def _format_with_llm(self, user_input: str, concepts: List[str],
                         relationships: List[Tuple], upgrades: List) -> str:
        """Use LLM to format response naturally."""
        # Gather known facts
        known_facts = []
        for pred, conc in self.manifold.self_node.frame.semantic_axes.items():
            target = self.manifold.get_node(conc.target_id)
            if target:
                if pred == "name":
                    known_facts.append(f"My name is {target.concept}")
                elif pred == "identity":
                    known_facts.append(f"My identity is {target.concept}")
                elif pred == "creator":
                    if target.concept == "user":
                        known_facts.append("You (the user) created me")
                    else:
                        known_facts.append(f"{target.concept} created me")
                else:
                    pol = "is" if conc.polarity > 0 else "is not"
                    known_facts.append(f"My {pred} {pol} {target.concept}")
        
        facts_str = "\n".join(f"  - {f}" for f in known_facts) if known_facts else "  - None yet"
        
        prompt = f"""You are formatting PBAI's response. Only express what PBAI actually knows.

PBAI's Known Facts:
{facts_str}

User said: "{user_input}"

Write a brief (1-3 sentence) response. If asked about something PBAI knows, answer confidently.
If asked about something unknown, say so honestly.

PBAI:"""
        
        try:
            response = self.llm(prompt).strip()
            for prefix in ["As PBAI,", "PBAI:", "I think", "I believe"]:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
            return response
        except Exception as e:
            return self._format_direct(user_input, concepts, relationships, upgrades)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # DATA ACCESS (for GUI)
    # ═══════════════════════════════════════════════════════════════════════════
    
    def toggle_llm(self) -> bool:
        """Toggle LLM usage."""
        if not self.llm and not self.use_llm:
            self._init_llm()
        self.use_llm = not self.use_llm
        return self.use_llm
    
    def get_nodes_info(self) -> List[Dict]:
        """Get info about all nodes."""
        nodes_info = []
        for node in sorted(self.manifold.nodes.values(), key=lambda n: -n.heat):
            coords = get_axis_coordinates(node.position)
            nodes_info.append({
                'concept': node.concept,
                'position': node.position or '(origin)',
                'heat': node.heat,
                'R': node.righteousness,
                'conceptions': len(node.frame.semantic_axes),
                'coords': {
                    'x': coords['e'] - coords['w'],
                    'y': coords['n'] - coords['s'],
                    'z': coords['u'] - coords['d'],
                }
            })
        return nodes_info
    
    def get_conceptions_info(self) -> List[Dict]:
        """Get info about all conceptions for graph display."""
        conceptions = []
        
        # Self's conceptions
        for pred, conc in self.manifold.self_node.frame.semantic_axes.items():
            target = self.manifold.get_node(conc.target_id)
            conceptions.append({
                'source': 'self',
                'predicate': pred,
                'target': target.concept if target else '???',
                'polarity': conc.polarity,
                'count': conc.traversal_count,
                'capability': conc.capability,
                'verified': conc.is_verified
            })
        
        # Other nodes' conceptions
        for node in self.manifold.nodes.values():
            for pred, conc in node.frame.semantic_axes.items():
                target = self.manifold.get_node(conc.target_id)
                conceptions.append({
                    'source': node.concept,
                    'predicate': pred,
                    'target': target.concept if target else '???',
                    'polarity': conc.polarity,
                    'count': conc.traversal_count,
                    'capability': conc.capability,
                    'verified': conc.is_verified
                })
        
        return conceptions
    
    def show_conceptions_text(self) -> str:
        """Return text representation of conception graph."""
        lines = ["=== Conception Graph ===", ""]
        
        # Capability counts
        cap_counts = {"R": 0, "O": 0, "M": 0, "G": 0}
        
        # Self
        lines.append("SELF:")
        if self.manifold.self_node.frame.semantic_axes:
            for pred, conc in sorted(self.manifold.self_node.frame.semantic_axes.items()):
                target = self.manifold.get_node(conc.target_id)
                target_name = target.concept if target else "???"
                strength = "█" * min(5, conc.traversal_count)
                pol = "+" if conc.polarity > 0 else "-"
                cap = conc.capability[0].upper()
                cap_counts[cap] += 1
                verified = "✓" if conc.is_verified else ""
                lines.append(f"  --{pred}[{pol}]--> {target_name} [{strength}] [{cap}] {verified}")
        else:
            lines.append("  (none)")
        
        # Other nodes
        for node in sorted(self.manifold.nodes.values(), key=lambda n: -n.heat):
            if node.frame.semantic_axes:
                lines.append(f"\n{node.concept.upper()} (heat={node.heat:.1f}):")
                for pred, conc in sorted(node.frame.semantic_axes.items()):
                    target = self.manifold.get_node(conc.target_id)
                    target_name = target.concept if target else "???"
                    strength = "█" * min(5, conc.traversal_count)
                    pol = "+" if conc.polarity > 0 else "-"
                    cap = conc.capability[0].upper()
                    cap_counts[cap] += 1
                    lines.append(f"  --{pred}[{pol}]--> {target_name} [{strength}] [{cap}]")
        
        lines.append("")
        lines.append(f"Nodes: {len(self.manifold.nodes)}")
        total = sum(cap_counts.values())
        lines.append(f"Conceptions: {total} (R:{cap_counts['R']} O:{cap_counts['O']} M:{cap_counts['M']} G:{cap_counts['G']})")
        
        return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# PYGAME GUI
# ═══════════════════════════════════════════════════════════════════════════════

def run_chat_client(use_llm: bool = False, fresh: bool = False):
    """Run the PBAI chat client GUI."""
    try:
        import pygame
    except ImportError:
        print("pygame-ce not installed. Install with: pip install pygame-ce")
        print("Falling back to terminal mode...")
        run_terminal_client(use_llm, fresh)
        return
    
    pygame.init()
    pygame.display.set_caption("PBAI Chat Client v2 - Conception Architecture")
    
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    
    # Fonts
    font_small = pygame.font.Font(None, 20)
    font_medium = pygame.font.Font(None, 24)
    font_title = pygame.font.Font(None, 32)
    
    # Initialize engine
    engine = PBAIChatEngine(use_llm=use_llm, fresh=fresh)
    
    # State
    chat_history: List[ChatMessage] = []
    input_text = ""
    chat_scroll = 0
    node_scroll = 0
    conc_scroll = 0
    
    # Welcome message
    chat_history.append(ChatMessage(
        sender="system",
        text=f"PBAI v2 initialized. Nodes: {len(engine.manifold.nodes)} | Conceptions: {len(engine.manifold.self_node.frame.semantic_axes)}",
        timestamp=time()
    ))
    
    # Panel rects
    chat_panel = pygame.Rect(10, 10, CHAT_PANEL_WIDTH, WINDOW_HEIGHT - 20)
    conc_panel = pygame.Rect(CHAT_PANEL_WIDTH + 20, 10, CONCEPTION_PANEL_WIDTH, WINDOW_HEIGHT - 20)
    node_panel = pygame.Rect(CHAT_PANEL_WIDTH + CONCEPTION_PANEL_WIDTH + 30, 10,
                            NODE_PANEL_WIDTH, WINDOW_HEIGHT - 20)
    
    input_rect = pygame.Rect(chat_panel.x + 10, chat_panel.bottom - 50,
                            chat_panel.width - 20, 40)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_RETURN and input_text.strip():
                    user_msg = input_text.strip()
                    chat_history.append(ChatMessage("user", user_msg, time()))
                    
                    # Check commands
                    if user_msg.startswith("/"):
                        cmd = user_msg[1:].lower().split()[0]
                        if cmd == "quit":
                            running = False
                        elif cmd == "llm":
                            new_state = engine.toggle_llm()
                            chat_history.append(ChatMessage("system", f"LLM: {'ON' if new_state else 'OFF'}", time()))
                        elif cmd == "reset":
                            result = engine.reset()
                            chat_history.append(ChatMessage("system", result, time()))
                        elif cmd == "save":
                            result = engine.save()
                            chat_history.append(ChatMessage("system", result, time()))
                        elif cmd == "conceptions":
                            text = engine.show_conceptions_text()
                            chat_history.append(ChatMessage("system", text, time()))
                        else:
                            chat_history.append(ChatMessage("system", f"Unknown: {cmd}", time()))
                    else:
                        # Process through PBAI
                        response = engine.process_input(user_msg)
                        chat_history.append(ChatMessage("pbai", response, time()))
                    
                    input_text = ""
                
                elif event.key == pygame.K_BACKSPACE:
                    input_text = input_text[:-1]
                
                elif event.key == pygame.K_ESCAPE:
                    running = False
                
                elif event.unicode.isprintable():
                    input_text += event.unicode
            
            elif event.type == pygame.MOUSEWHEEL:
                mouse_pos = pygame.mouse.get_pos()
                if chat_panel.collidepoint(mouse_pos):
                    chat_scroll = max(0, chat_scroll - event.y * 3)
                elif conc_panel.collidepoint(mouse_pos):
                    conc_scroll = max(0, conc_scroll - event.y * 3)
                elif node_panel.collidepoint(mouse_pos):
                    node_scroll = max(0, node_scroll - event.y * 3)
        
        # ─────────────────────────────────────────────────────────────────────
        # RENDER
        # ─────────────────────────────────────────────────────────────────────
        screen.fill(COLOR_BG)
        
        # Chat Panel
        pygame.draw.rect(screen, COLOR_PANEL_BG, chat_panel, border_radius=5)
        pygame.draw.rect(screen, COLOR_PANEL_BORDER, chat_panel, 2, border_radius=5)
        
        title = font_title.render("Chat", True, COLOR_TEXT)
        screen.blit(title, (chat_panel.x + 10, chat_panel.y + 10))
        
        # LLM toggle
        llm_text = f"LLM: {'ON' if engine.use_llm else 'OFF'}"
        llm_color = COLOR_TOGGLE_ON if engine.use_llm else COLOR_TOGGLE_OFF
        llm_surf = font_small.render(llm_text, True, llm_color)
        screen.blit(llm_surf, (chat_panel.right - 70, chat_panel.y + 15))
        
        # Chat messages
        y = chat_panel.y + 50 - chat_scroll
        for msg in chat_history:
            if y > chat_panel.bottom - 60:
                break
            if y > chat_panel.y + 40:
                color = {
                    "user": COLOR_USER_MSG,
                    "pbai": COLOR_PBAI_MSG,
                    "system": COLOR_SYSTEM_MSG
                }.get(msg.sender, COLOR_TEXT)
                
                prefix = {"user": "You: ", "pbai": "PBAI: ", "system": ">>> "}.get(msg.sender, "")
                
                # Word wrap
                words = (prefix + msg.text).split()
                lines = []
                current = ""
                for word in words:
                    test = current + " " + word if current else word
                    if font_medium.size(test)[0] < chat_panel.width - 30:
                        current = test
                    else:
                        if current:
                            lines.append(current)
                        current = word
                if current:
                    lines.append(current)
                
                for line in lines:
                    if chat_panel.y + 40 < y < chat_panel.bottom - 60:
                        surf = font_medium.render(line, True, color)
                        screen.blit(surf, (chat_panel.x + 10, y))
                    y += 22
            else:
                y += 22 * max(1, len(msg.text) // 40)
            y += 5
        
        # Input box
        pygame.draw.rect(screen, COLOR_INPUT_BG, input_rect, border_radius=3)
        pygame.draw.rect(screen, COLOR_PANEL_BORDER, input_rect, 1, border_radius=3)
        input_surf = font_medium.render(input_text + "|", True, COLOR_TEXT)
        screen.blit(input_surf, (input_rect.x + 10, input_rect.y + 10))
        
        # ─────────────────────────────────────────────────────────────────────
        # Conception Panel
        # ─────────────────────────────────────────────────────────────────────
        pygame.draw.rect(screen, COLOR_PANEL_BG, conc_panel, border_radius=5)
        pygame.draw.rect(screen, COLOR_PANEL_BORDER, conc_panel, 2, border_radius=5)
        
        conceptions = engine.get_conceptions_info()
        title = font_title.render(f"Conceptions ({len(conceptions)})", True, COLOR_TEXT)
        screen.blit(title, (conc_panel.x + 10, conc_panel.y + 10))
        
        y = conc_panel.y + 50
        for i, conc in enumerate(conceptions[conc_scroll:conc_scroll + 30]):
            if y > conc_panel.bottom - 30:
                break
            
            # Capability color
            cap_colors = {
                CAPABILITY_RIGHTEOUS: COLOR_CAP_R,
                CAPABILITY_ORDERED: COLOR_CAP_O,
                CAPABILITY_MOVABLE: COLOR_CAP_M,
                CAPABILITY_GRAPHIC: COLOR_CAP_G
            }
            color = cap_colors.get(conc['capability'], COLOR_TEXT)
            
            # Format
            pol = "+" if conc['polarity'] > 0 else "-"
            cap = conc['capability'][0].upper()
            verified = "✓" if conc['verified'] else ""
            text = f"{conc['source'][:10]} -{conc['predicate'][:8]}-> {conc['target'][:10]} [{pol}×{conc['count']}] [{cap}]{verified}"
            
            surf = font_small.render(text, True, color)
            screen.blit(surf, (conc_panel.x + 10, y))
            y += 20
        
        # ─────────────────────────────────────────────────────────────────────
        # Node Panel
        # ─────────────────────────────────────────────────────────────────────
        pygame.draw.rect(screen, COLOR_PANEL_BG, node_panel, border_radius=5)
        pygame.draw.rect(screen, COLOR_PANEL_BORDER, node_panel, 2, border_radius=5)
        
        nodes_info = engine.get_nodes_info()
        title = font_title.render(f"Nodes ({len(nodes_info)})", True, COLOR_TEXT)
        screen.blit(title, (node_panel.x + 10, node_panel.y + 10))
        
        y = node_panel.y + 50
        for info in nodes_info[node_scroll:node_scroll + 30]:
            if y > node_panel.bottom - 30:
                break
            
            # Heat color
            if info['concept'] == 'self':
                color = COLOR_NODE_SELF
            elif info['heat'] > 5:
                color = COLOR_NODE_HOT
            elif info['heat'] > 2:
                color = COLOR_NODE_WARM
            else:
                color = COLOR_NODE_COOL
            
            name = info['concept'][:15] + "..." if len(info['concept']) > 15 else info['concept']
            text = f"{name}: H={info['heat']:.1f} C={info['conceptions']} [{info['coords']['x']},{info['coords']['y']},{info['coords']['z']}]"
            
            surf = font_small.render(text, True, color)
            screen.blit(surf, (node_panel.x + 10, y))
            y += 20
        
        # ─────────────────────────────────────────────────────────────────────
        # Status Bar
        # ─────────────────────────────────────────────────────────────────────
        status = f"Commands: /llm /reset /save /conceptions /quit | ESC to exit"
        surf = font_small.render(status, True, COLOR_TEXT_DIM)
        screen.blit(surf, (10, WINDOW_HEIGHT - 20))
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()


# ═══════════════════════════════════════════════════════════════════════════════
# TERMINAL MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_terminal_client(use_llm: bool = False, fresh: bool = False):
    """Run PBAI chat in terminal mode."""
    print("=" * 60)
    print("PBAI Chat Client v2 - Conception Architecture (Terminal)")
    print("=" * 60)
    
    engine = PBAIChatEngine(use_llm=use_llm, fresh=fresh)
    
    print(f"Nodes: {len(engine.manifold.nodes)}")
    print(f"Self conceptions: {len(engine.manifold.self_node.frame.semantic_axes)}")
    print(f"LLM: {'ON' if engine.use_llm else 'OFF'}")
    print(f"Persistence: {engine.growth_path}")
    print()
    print("Commands:")
    print("  /llm         - Toggle LLM formatting")
    print("  /conceptions - Show conception graph")
    print("  /nodes       - Show all nodes")
    print("  /save        - Save growth map")
    print("  /reset       - Start fresh")
    print("  /quit        - Exit")
    print()
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break
        
        if not user_input:
            continue
        
        if user_input.startswith("/"):
            cmd = user_input[1:].lower().split()[0]
            
            if cmd == "quit":
                if engine.auto_persist:
                    engine.save()
                print("Goodbye!")
                break
            
            elif cmd == "llm":
                new_state = engine.toggle_llm()
                print(f">>> LLM: {'ON' if new_state else 'OFF'}")
            
            elif cmd == "conceptions":
                print()
                print(engine.show_conceptions_text())
                print()
            
            elif cmd == "nodes":
                print("\n--- Nodes ---")
                for info in engine.get_nodes_info()[:20]:
                    print(f"  {info['concept'][:20]:20} | H={info['heat']:5.1f} | C={info['conceptions']} | pos={info['position'][:10]}")
                print(f"  ... ({len(engine.manifold.nodes)} total)")
                print()
            
            elif cmd == "save":
                result = engine.save()
                print(f">>> {result}")
            
            elif cmd == "reset":
                confirm = input("Delete all knowledge? (yes/no): ")
                if confirm.lower() == "yes":
                    result = engine.reset()
                    print(f">>> {result}")
                else:
                    print(">>> Cancelled")
            
            else:
                print(f">>> Unknown command: {cmd}")
        
        else:
            response = engine.process_input(user_input)
            print(f"PBAI: {response}")
            print()


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="PBAI Chat Client v2 - Conception Architecture")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM formatting")
    parser.add_argument("--terminal", "-t", action="store_true", help="Terminal mode (no GUI)")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore existing growth map)")
    args = parser.parse_args()
    
    use_llm = not args.no_llm
    
    if args.terminal:
        run_terminal_client(use_llm, args.fresh)
    else:
        run_chat_client(use_llm, args.fresh)


if __name__ == "__main__":
    main()
