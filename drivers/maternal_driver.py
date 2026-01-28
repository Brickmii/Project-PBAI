"""
Maternal Driver - PBAI learns language through guided teaching

PBAI starts with no language. When it can't respond confidently,
it asks "mom" (Qwen) for guidance. Over time, PBAI builds its own
thermal lexicon and needs mom less and less.

ARCHITECTURE:
    Word = Righteous Frame (R=1.0)
    Properties = Proper Frames (R=0.0-0.9)
    Connections = Semantic Axes
    
    "dog" (R=1.0, word frame)
      ├── axis "is" → "animal"
      ├── axis "has" → "fur"
      ├── axis "does" → "bark"
      └── heat = familiarity/usage

LEARNING STAGES:
    Infant: No words, zero confidence → full Qwen guidance
    Child: Some words, partial responses → Qwen fills gaps
    Adolescent: Growing lexicon → deliberation, sometimes rejects Qwen
    Adult: Thermal mass exceeds Qwen's context → PBAI leads

CLOCK SYNC:
    Each conversation turn = one tick
    Words heat up with use
    Unused words cool over time
    Conscience tracks motion calendar
"""

import logging
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from time import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Manifold, K
from core.nodes import Node
from core.clock_node import Clock
from core.decision_node import DecisionNode

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

# Confidence thresholds
CONFIDENCE_INFANT = 0.1      # Below this = know nothing, full Qwen
CONFIDENCE_CHILD = 0.4       # Below this = partial knowledge, Qwen helps
CONFIDENCE_ADOLESCENT = 0.7  # Below this = deliberate with Qwen
CONFIDENCE_ADULT = 0.9       # Above this = respond independently

# NO STOPWORDS - the manifold absorbs everything
# Heat and connections determine salience, not pre-filtering
# A child learns "the", "is", "my", "your" - they're grammar, not garbage

# Relationship patterns for extraction
RELATION_PATTERNS = [
    (r'(\w+)\s+is\s+a\s+(\w+)', 'is_a'),           # X is a Y
    (r'(\w+)\s+are\s+(\w+)', 'are'),                # X are Y
    (r'(\w+)\s+has\s+(\w+)', 'has'),                # X has Y
    (r'(\w+)\s+have\s+(\w+)', 'has'),               # X have Y
    (r'(\w+)\s+can\s+(\w+)', 'can'),                # X can Y
    (r'(\w+)\s+is\s+(\w+)', 'is'),                  # X is Y
    (r'(\w+)\s+means\s+(\w+)', 'means'),            # X means Y
    (r'(\w+)\s+like\s+(\w+)', 'like'),              # X like Y
    (r'(\w+)\s+called\s+(\w+)', 'called'),          # X called Y
]


@dataclass
class ConversationTurn:
    """A single turn in conversation."""
    user_input: str
    pbai_response: str
    qwen_consulted: bool
    qwen_response: Optional[str]
    confidence: float
    accepted_qwen: Optional[bool]  # None if not consulted, True/False if deliberated
    timestamp: float


# ═══════════════════════════════════════════════════════════════════════════════
# MATERNAL DRIVER
# ═══════════════════════════════════════════════════════════════════════════════

class MaternalDriver:
    """
    PBAI learns language through maternal (Qwen) guidance.
    
    Same interface pattern as maze/blackjack:
    - Clock sync for time
    - DecisionNode for word selection
    - Psychology for confidence/deliberation
    """
    
    DRIVER_ID = "maternal"
    SUPPORTED_ACTIONS = ['respond', 'ask_qwen', 'accept', 'reject', 'deliberate']
    
    def __init__(self, manifold: Manifold, qwen_model: str = "Qwen/Qwen2.5-0.5B-Instruct",
                 use_qwen: bool = True):
        """
        Initialize MaternalDriver.
        
        Args:
            manifold: PBAI manifold
            qwen_model: Qwen model to use as "mother"
                - "Qwen/Qwen2.5-0.5B-Instruct" (default, Pi-friendly, ~1GB)
                - "Qwen/Qwen2.5-1.5B-Instruct" (better quality, ~3GB)
                - "Qwen/Qwen2.5-3B-Instruct" (best quality, ~6GB)
            use_qwen: If False, runs pure thermal (no mother)
        """
        self.manifold = manifold
        self.qwen_model = qwen_model
        self.use_qwen = use_qwen  # Can disable Qwen for pure thermal testing
        
        # Clock for perception routing
        self._clock: Optional[Clock] = None
        
        # DecisionNode for action routing
        self._decision_node: Optional[DecisionNode] = None
        
        # Qwen (mom) - initialized lazily
        self._qwen = None
        self._qwen_tokenizer = None
        
        # Task frame
        self.task_frame: Optional[Node] = None
        
        # Conversation history
        self.history: List[ConversationTurn] = []
        
        # Stats
        self.total_turns = 0
        self.qwen_consultations = 0
        self.qwen_rejections = 0
        self.words_learned = 0
        
        # Current state
        self._current_input: str = ""
        self._current_context: Dict = {}
        
        # Initialize
        self._init_task_frame()
        self._init_core_words()
    
    def _get_clock(self) -> Clock:
        """Get or create Clock."""
        if self._clock is None:
            self._clock = Clock(self.manifold)
        return self._clock
    
    def _get_decision_node(self) -> DecisionNode:
        """Get or create DecisionNode."""
        if self._decision_node is None:
            self._decision_node = DecisionNode(self.manifold)
        return self._decision_node
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INITIALIZATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _init_task_frame(self):
        """Create the language task frame."""
        self.task_frame = self.manifold.get_node_by_concept("language")
        if not self.task_frame:
            self.task_frame = Node(
                concept="language",
                position="u",
                heat=K,
                polarity=1,
                existence="actual",
                righteousness=1.0,  # Righteous frame
                order=1
            )
            self.manifold.add_node(self.task_frame)
            logger.info("Task frame created: language (R=1.0)")
    
    def _init_core_words(self):
        """Initialize core vocabulary if not present."""
        # These are fundamental words PBAI starts with
        core_words = [
            # Self-reference
            ("i", "pronoun", 1),
            ("me", "pronoun", 1),
            ("my", "pronoun", 1),
            # Basic concepts
            ("yes", "affirmation", 1),
            ("no", "negation", -1),
            ("hello", "greeting", 1),
            ("help", "request", 1),
            # Question words
            ("what", "question", 0),
            ("why", "question", 0),
            ("how", "question", 0),
        ]
        
        for word, category, polarity in core_words:
            self._get_or_create_word(word, category=category, polarity=polarity)
    
    def _init_qwen(self):
        """Initialize Qwen (mom) for guidance."""
        if self._qwen is not None:
            return
        
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            logger.info(f"Loading mom (Qwen): {self.qwen_model}...")
            
            self._qwen_tokenizer = AutoTokenizer.from_pretrained(self.qwen_model)
            
            # Check if CUDA available, otherwise CPU
            if torch.cuda.is_available():
                self._qwen = AutoModelForCausalLM.from_pretrained(
                    self.qwen_model,
                    device_map="auto",
                    torch_dtype=torch.float16
                )
                logger.info("Mom loaded on GPU")
            else:
                # Pi-friendly: CPU with float32
                self._qwen = AutoModelForCausalLM.from_pretrained(
                    self.qwen_model,
                    device_map="cpu",
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                logger.info("Mom loaded on CPU (Pi mode)")
            
            logger.info("Mom is ready")
        except Exception as e:
            logger.warning(f"Could not load Qwen: {e}")
            logger.warning("PBAI will operate without maternal guidance")
            self._qwen = None
    
    # ═══════════════════════════════════════════════════════════════════════════
    # WORD MANAGEMENT
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _get_word_node(self, word: str) -> Optional[Node]:
        """Get word node if it exists."""
        concept = f"word_{word.lower()}"
        return self.manifold.get_node_by_concept(concept)
    
    def _get_or_create_word(self, word: str, category: str = "unknown", 
                           polarity: int = 0) -> Node:
        """Get or create a word node (righteous frame)."""
        word = word.lower().strip()
        concept = f"word_{word}"
        
        node = self.manifold.get_node_by_concept(concept)
        if node:
            # Heat up existing word (familiarity increases)
            node.add_heat(K * 0.1)
            return node
        
        # Create new word node
        # Position under language task frame
        existing = [n for n in self.manifold.nodes.values() 
                   if n.position.startswith(self.task_frame.position + "c")]
        idx = len(existing)
        position = f"{self.task_frame.position}c{idx}"
        
        node = Node(
            concept=concept,
            position=position,
            heat=K,
            polarity=polarity,
            existence="actual",
            righteousness=1.0,  # Words are righteous frames
            order=2
        )
        self.manifold.add_node(node)
        
        # Add category as property axis
        if category != "unknown":
            cat_node = self._get_or_create_property(category)
            node.add_axis("category", cat_node.id)
        
        # ═══════════════════════════════════════════════════════════════════
        # CONNECT TO PSYCHOLOGY - words integrate with PBAI's mind
        # ═══════════════════════════════════════════════════════════════════
        
        # IDENTITY: "I know this word exists"
        self.manifold.update_identity(concept, heat_delta=K * 0.1, known=True)
        
        # EGO: "I can use this word" (pattern for word usage)
        self.manifold.update_ego(f"use_{word}", success=True, heat_delta=K * 0.05)
        
        # CONSCIENCE: "This word is validated" (learned from mom or confirmed)
        self.manifold.validate_conscience(concept, confirmed=True)
        
        # Special case: "pbai" is PBAI's own name - connect to Self!
        if word == "pbai" and self.manifold.self_node:
            self.manifold.self_node.add_axis("name", node.id)
            node.add_axis("is_self", self.manifold.self_node.id)
            logger.info("Connected 'pbai' to Self as name")
        
        self.words_learned += 1
        logger.debug(f"Learned word: {word} ({category})")
        
        return node
    
    def _get_or_create_property(self, prop: str) -> Node:
        """Get or create a property node (proper frame)."""
        prop = prop.lower().strip()
        concept = f"prop_{prop}"
        
        node = self.manifold.get_node_by_concept(concept)
        if node:
            return node
        
        # Create property node (proper frame, under language)
        existing = [n for n in self.manifold.nodes.values() 
                   if n.position.startswith(self.task_frame.position + "p")]
        idx = len(existing)
        position = f"{self.task_frame.position}p{idx}"
        
        node = Node(
            concept=concept,
            position=position,
            heat=K * 0.5,
            polarity=0,
            existence="actual",
            righteousness=0.5,  # Proper frame
            order=3
        )
        self.manifold.add_node(node)
        
        return node
    
    def _connect_words(self, word1: str, relation: str, word2: str):
        """Create semantic axis between words and track in Ego."""
        node1 = self._get_or_create_word(word1)
        node2 = self._get_or_create_word(word2)
        
        # Add axis from word1 to word2
        axis = node1.add_axis(relation, node2.id)
        axis.strengthen()  # Increment traversal count
        
        # Track this relationship pattern in Ego - "I learned X relates to Y"
        pattern = f"rel_{word1}_{relation}_{word2}"
        self.manifold.update_ego(pattern, success=True, heat_delta=K * 0.05)
        
        logger.debug(f"Connected: {word1} --{relation}--> {word2}")
    
    def _heat_word(self, word: str, amount: float = None):
        """Add heat to a word node (reinforcement) and strengthen Ego."""
        node = self._get_word_node(word)
        if node:
            heat = amount if amount is not None else K * 0.1
            node.add_heat(heat)
            
            # Strengthen Ego - "I successfully used this word"
            self.manifold.update_ego(f"use_{word}", success=True, heat_delta=heat * 0.5)
            
            logger.debug(f"Heated word '{word}' by {heat:.2f}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEXT PROCESSING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _extract_words(self, text: str) -> List[str]:
        """Extract meaningful words from text."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        # Filter stopwords but keep them for learning basic structure
        return [w for w in words if len(w) > 1]
    
    def _extract_content_words(self, text: str) -> List[str]:
        """Extract all words from text - no filtering, manifold absorbs everything."""
        words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
        return [w for w in words if len(w) > 1]  # Only filter single letters
    
    def _extract_relations(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract word relationships from text."""
        relations = []
        text_lower = text.lower()
        
        for pattern, relation in RELATION_PATTERNS:
            matches = re.findall(pattern, text_lower)
            for match in matches:
                if len(match) == 2:
                    word1, word2 = match
                    # No filtering - learn all relations
                    relations.append((word1, relation, word2))
        
        return relations
    
    def _learn_from_text(self, text: str):
        """Learn words and relationships from text."""
        # Learn all words
        words = self._extract_words(text)
        for word in words:
            self._get_or_create_word(word)
        
        # Learn relationships
        relations = self._extract_relations(text)
        for word1, relation, word2 in relations:
            self._connect_words(word1, relation, word2)
        
        # Connect adjacent content words (co-occurrence)
        content_words = self._extract_content_words(text)
        for i in range(len(content_words) - 1):
            self._connect_words(content_words[i], "adjacent", content_words[i + 1])
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIDENCE CALCULATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _calculate_confidence(self, words: List[str]) -> float:
        """
        Calculate confidence for responding to these words.
        
        Based on:
        - How many words we know
        - How hot those words are
        - How connected they are
        """
        if not words:
            return 0.0
        
        known_count = 0
        total_heat = 0.0
        total_connections = 0
        
        for word in words:
            node = self._get_word_node(word)
            if node:
                known_count += 1
                total_heat += node.heat
                # Count semantic connections
                if hasattr(node, 'frame') and node.frame:
                    total_connections += len(node.frame.axes)
        
        if known_count == 0:
            return 0.0
        
        # Factors
        coverage = known_count / len(words)
        avg_heat = total_heat / known_count / K  # Normalize by K
        avg_connections = total_connections / known_count / 5  # Normalize (5 = expected connections)
        
        # Weighted combination
        confidence = (
            coverage * 0.4 +
            min(1.0, avg_heat) * 0.3 +
            min(1.0, avg_connections) * 0.3
        )
        
        return min(1.0, confidence)
    
    def _get_development_stage(self, confidence: float) -> str:
        """
        Get current development stage based on vocabulary and independence.
        
        Note: This is for DISPLAY purposes. The actual decision to ask Qwen
        is based on whether there are unknown words, not the stage.
        """
        vocab_size = self.get_vocabulary_size()
        independence = self.qwen_rejections / max(1, self.qwen_consultations) if self.qwen_consultations > 0 else 0
        
        if vocab_size < 50:
            return "infant"
        elif vocab_size < 200:
            return "child"
        elif vocab_size < 1000 or independence < 0.3:
            return "adolescent"
        else:
            return "adult"
    
    # ═══════════════════════════════════════════════════════════════════════════
    # QWEN INTERACTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _build_knowledge_context(self, input_words: List[str], unknown_words: List[str]) -> Tuple[str, str]:
        """
        Build context of what PBAI knows vs doesn't know.
        
        Returns: (known_summary, unknown_summary)
        """
        known_facts = []
        known_words = [w for w in input_words if w not in unknown_words]
        
        for word in known_words:
            node = self._get_word_node(word)
            if not node or not hasattr(node, 'frame') or not node.frame:
                continue
            
            # Get meaningful relationships this word has
            for axis_name, axis in node.frame.axes.items():
                if axis_name in ['adjacent', 'context', 'category']:
                    continue  # Skip generic axes
                
                target = self.manifold.get_node(axis.target_id)
                if not target:
                    continue
                
                if target.concept.startswith("word_"):
                    target_word = target.concept.replace("word_", "")
                    known_facts.append(f"{word} {axis_name} {target_word}")
                elif target.concept == "self":
                    known_facts.append(f"{word} is self (PBAI)")
        
        # Check Self node for identity info
        if self.manifold.self_node:
            for axis_name, axis in self.manifold.self_node.frame.axes.items():
                if axis_name == "name":
                    target = self.manifold.get_node(axis.target_id)
                    if target:
                        known_facts.append(f"my name is {target.concept.replace('word_', '')}")
        
        known_summary = "; ".join(known_facts[:10]) if known_facts else "very little"
        unknown_summary = ", ".join(unknown_words) if unknown_words else "nothing new"
        
        return known_summary, unknown_summary
    
    def _ask_qwen(self, user_input: str, known_words: List[str] = None, 
                  unknown_words: List[str] = None) -> str:
        """
        Ask Qwen (mom) for guidance, providing what PBAI already knows.
        
        Key: Tell Qwen what we know so it builds on existing knowledge,
        and what we don't know so it fills those gaps specifically.
        """
        if not self.use_qwen:
            return "[Mom disabled - responding from thermal state only]"
        
        if self._qwen is None:
            self._init_qwen()
        
        if self._qwen is None:
            return "[Mom unavailable - responding from what I know]"
        
        self.qwen_consultations += 1
        
        # Build knowledge context
        if known_words and unknown_words:
            known_summary, unknown_summary = self._build_knowledge_context(
                known_words + unknown_words, unknown_words
            )
            
            prompt = f"""You are teaching a child named PBAI who is learning language.
PBAI is an artificial intelligence. The human talking to PBAI is Ian, PBAI's creator.

PBAI already knows: {known_summary}

PBAI was asked: "{user_input}"

PBAI doesn't know these words: {unknown_summary}

Respond AS PBAI (first person). Explain the unknown words in context of what PBAI already knows.
Keep it brief and simple. Build on existing knowledge, don't contradict it.
Remember: PBAI is the AI, Ian is the human creator."""

        elif unknown_words:
            prompt = f"""You are teaching a child named PBAI who is learning language from scratch.
PBAI is an artificial intelligence created by Ian (a human).

PBAI was asked: "{user_input}"

PBAI doesn't know these words: {", ".join(unknown_words)}

Respond AS PBAI (first person). Use very simple words. 
Remember: PBAI is the AI, Ian is the human creator."""

        else:
            prompt = f"""You are teaching a child named PBAI who is learning language.
PBAI is an artificial intelligence created by Ian (a human).

PBAI was asked: "{user_input}"

Help PBAI formulate a simple response. Respond AS PBAI (first person).
Remember: PBAI is the AI, Ian is the human creator."""
        
        try:
            messages = [{"role": "user", "content": prompt}]
            text = self._qwen_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self._qwen_tokenizer(text, return_tensors="pt").to(self._qwen.device)
            
            outputs = self._qwen.generate(
                **inputs,
                max_new_tokens=2048,  # Qwen2.5-0.5B supports up to 32K context
                do_sample=True,
                temperature=0.7,
                pad_token_id=self._qwen_tokenizer.eos_token_id
            )
            
            response = self._qwen_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
        
        except Exception as e:
            logger.error(f"Qwen error: {e}")
            return "[Mom had trouble understanding - try again]"
    
    def _deliberate_with_qwen(self, user_input: str, pbai_response: str, 
                              qwen_suggestion: str) -> Tuple[str, bool]:
        """
        Deliberate between PBAI's response and Qwen's suggestion.
        
        Returns: (final_response, accepted_qwen)
        """
        # Calculate thermal support for each response
        pbai_words = self._extract_content_words(pbai_response)
        qwen_words = self._extract_content_words(qwen_suggestion)
        
        pbai_heat = sum(
            self._get_word_node(w).heat if self._get_word_node(w) else 0 
            for w in pbai_words
        )
        qwen_heat = sum(
            self._get_word_node(w).heat if self._get_word_node(w) else 0 
            for w in qwen_words
        )
        
        # PBAI's thermal evidence vs Qwen's suggestion
        # As PBAI grows, it trusts its own heat more
        pbai_confidence = self.manifold.get_confidence()
        
        # Decision threshold shifts with development
        accept_threshold = 0.5 - (pbai_confidence * 0.3)  # More confident = lower threshold to reject
        
        if qwen_heat > pbai_heat * (1 + accept_threshold):
            # Qwen's response has more thermal support or PBAI is young
            logger.info(f"Deliberation: Accepting Qwen (qwen_heat={qwen_heat:.1f} > pbai_heat={pbai_heat:.1f})")
            return qwen_suggestion, True
        else:
            # PBAI's response has enough thermal support
            logger.info(f"Deliberation: Rejecting Qwen (pbai_heat={pbai_heat:.1f} >= qwen_heat={qwen_heat:.1f})")
            self.qwen_rejections += 1
            return pbai_response, False
    
    # ═══════════════════════════════════════════════════════════════════════════
    # RESPONSE GENERATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def _calculate_heat_budget(self, input_words: List[str]) -> Tuple[float, float, float]:
        """
        Calculate heat budget for response.
        
        Budget = Connected Cluster Heat / Input Cluster Heat
        
        Returns: (input_heat, connected_heat, budget_ratio)
        """
        input_heat = 0.0
        connected_heat = 0.0
        seen_connections = set()
        
        for word in input_words:
            node = self._get_word_node(word)
            if not node:
                continue
            
            input_heat += node.heat
            
            # Sum heat of all connected nodes (the cluster)
            if hasattr(node, 'frame') and node.frame:
                for axis_name, axis in node.frame.axes.items():
                    if axis.target_id in seen_connections:
                        continue
                    seen_connections.add(axis.target_id)
                    
                    target = self.manifold.get_node(axis.target_id)
                    if target:
                        connected_heat += target.heat
        
        # Also check Self connections
        if self.manifold.self_node:
            for axis_name, axis in self.manifold.self_node.frame.axes.items():
                if axis.target_id not in seen_connections:
                    target = self.manifold.get_node(axis.target_id)
                    if target:
                        connected_heat += target.heat
                        seen_connections.add(axis.target_id)
        
        # Budget ratio (always positive, higher = more to say)
        budget_ratio = connected_heat / max(input_heat, K * 0.1)
        
        return input_heat, connected_heat, budget_ratio
    
    def _check_self_for_answer(self, input_words: List[str]) -> Optional[str]:
        """
        Check if this is a question about Self (your/you) and answer from Self node.
        
        "What is your name?" → check Self.name axis → "pbai"
        """
        self_words = {'your', 'you', 'yourself'}
        has_self_reference = any(w in self_words for w in input_words)
        
        if not has_self_reference or not self.manifold.self_node:
            return None
        
        # What are they asking about Self?
        question_words = {'what', 'who', 'how', 'why', 'where', 'when', 'which', 
                         'is', 'are', 'do', 'does', 'can', 'tell'}
        subject_words = [w for w in input_words 
                        if w not in question_words and w not in self_words]
        
        if not subject_words:
            return None
        
        # Check if Self has an axis matching the subject
        for subject in subject_words:
            axis = self.manifold.self_node.get_axis(subject)
            if axis:
                target = self.manifold.get_node(axis.target_id)
                if target:
                    target_name = target.concept.replace("word_", "").replace("prop_", "")
                    return f"my {subject} is {target_name}"
        
        return None
    
    def _generate_thermal_response(self, input_words: List[str]) -> str:
        """
        Generate response from thermal state by traversing axes.
        
        Key principles:
        1. Check Self node first for "your/you" questions
        2. DON'T echo input words - follow connections to find what they POINT TO
        3. Heat budget = connected heat / input heat (determines response length)
        4. Speaking COSTS heat (deducted from word nodes)
        5. Stop when budget exhausted or 5/6 threshold reached
        """
        from core.node_constants import CONFIDENCE_EXPLOIT_THRESHOLD, K  # 5/6
        
        if not input_words:
            return "..."
        
        # ═══════════════════════════════════════════════════════════════════
        # FIRST: Check if asking about Self (your/you questions)
        # ═══════════════════════════════════════════════════════════════════
        self_answer = self._check_self_for_answer(input_words)
        if self_answer:
            return self_answer
        
        # ═══════════════════════════════════════════════════════════════════
        # CALCULATE HEAT BUDGET
        # ═══════════════════════════════════════════════════════════════════
        input_heat, connected_heat, budget_ratio = self._calculate_heat_budget(input_words)
        
        if input_heat == 0:
            return "..."
        
        # Budget determines how much we can say
        # High ratio = lots of connections = can say a lot
        # Low ratio = few connections = brief response
        speech_budget = connected_heat * CONFIDENCE_EXPLOIT_THRESHOLD
        
        logger.debug(f"Heat budget: input={input_heat:.1f}, connected={connected_heat:.1f}, "
                    f"ratio={budget_ratio:.2f}, budget={speech_budget:.1f}")
        
        # ═══════════════════════════════════════════════════════════════════
        # FIND RELATIONS (prioritize meaningful axes)
        # ═══════════════════════════════════════════════════════════════════
        
        question_words = {'what', 'who', 'how', 'why', 'where', 'when', 'which', 
                        'is', 'are', 'do', 'does', 'can', 'tell'}
        subject_words = [w for w in input_words if w not in question_words]
        
        if not subject_words:
            return "..."
        
        meaningful_axes = ['is', 'is_a', 'are', 'has', 'can', 'does', 'means', 
                          'like', 'called', 'name', 'is_self', 'created']
        
        found_relations = []
        input_word_set = set(input_words)
        
        for word in subject_words:
            node = self._get_word_node(word)
            if not node or not hasattr(node, 'frame') or not node.frame:
                continue
            
            for axis_name, axis in node.frame.axes.items():
                target = self.manifold.get_node(axis.target_id)
                if not target:
                    continue
                
                # Get target word
                if target.concept.startswith("word_"):
                    target_word = target.concept.replace("word_", "")
                elif target.concept.startswith("prop_"):
                    target_word = target.concept.replace("prop_", "")
                elif target.concept == "self":
                    target_word = "pbai"
                else:
                    continue
                
                # Skip echoing input words
                if target_word in input_word_set:
                    continue
                
                # Weight by axis type
                weight = target.heat
                if axis_name in meaningful_axes:
                    weight *= 2.0
                if axis_name in ['adjacent', 'context']:
                    weight *= 0.3
                
                # Speech cost = fraction of target's heat
                speech_cost = target.heat * 0.1
                
                found_relations.append((axis_name, target_word, weight, word, target.heat, speech_cost))
        
        if not found_relations:
            if subject_words:
                return f"i know {subject_words[0]}"
            return "..."
        
        # Sort by weight (meaningful + hot first)
        found_relations.sort(key=lambda x: x[2], reverse=True)
        
        # ═══════════════════════════════════════════════════════════════════
        # BUILD RESPONSE (spend heat budget)
        # ═══════════════════════════════════════════════════════════════════
        
        spent_heat = 0.0
        response_parts = []
        used_targets = set()
        
        for relation, target, weight, source, raw_heat, speech_cost in found_relations:
            # Check budget
            if spent_heat + speech_cost > speech_budget:
                break  # Out of budget
            
            if target in used_targets:
                continue
            
            # Structure the fact
            if relation in ['is', 'is_a', 'are']:
                part = f"{source} is {target}"
            elif relation == 'has':
                part = f"{source} has {target}"
            elif relation == 'can':
                part = f"{source} can {target}"
            elif relation == 'name':
                part = f"my name is {target}"
            elif relation == 'is_self':
                part = f"i am {source}"
            elif relation == 'created':
                part = f"{source} was created by {target}"
            else:
                part = f"{source} {relation} {target}"
            
            response_parts.append(part)
            used_targets.add(target)
            spent_heat += speech_cost
            
            # COST HEAT for speaking (deduct from target node)
            target_node = self._get_word_node(target)
            if target_node and target_node.heat > speech_cost:
                target_node.heat -= speech_cost
                logger.debug(f"Speech cost: '{target}' -{speech_cost:.2f} heat")
        
        if not response_parts:
            return "..."
        
        logger.debug(f"Response used {spent_heat:.1f} of {speech_budget:.1f} budget")
        
        if len(response_parts) == 1:
            return response_parts[0]
        else:
            return " and ".join(response_parts)
    
    def _get_known_context(self, words: List[str]) -> str:
        """Get what PBAI knows about the input words."""
        context_lines = []
        
        for word in words:
            node = self._get_word_node(word)
            if node:
                connections = []
                if hasattr(node, 'frame') and node.frame:
                    for axis_name, axis in node.frame.axes.items():
                        target = self.manifold.get_node(axis.target_id)
                        if target:
                            target_name = target.concept.replace("word_", "").replace("prop_", "")
                            connections.append(f"{axis_name}:{target_name}")
                
                if connections:
                    context_lines.append(f"  {word}: {', '.join(connections[:5])}")
                else:
                    context_lines.append(f"  {word}: (known, no connections)")
        
        return "\n".join(context_lines) if context_lines else ""
    
    # ═══════════════════════════════════════════════════════════════════════════
    # MAIN INTERFACE
    # ═══════════════════════════════════════════════════════════════════════════
    
    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate response.
        
        Core logic:
        1. Extract content words from input
        2. Check which words are UNKNOWN (no node or low heat)
        3. If ANY unknown → ask Qwen the FULL question → learn in context
        4. If ALL known → respond from thermal state (traverse manifold)
        
        This is contextual learning - PBAI learns new concepts in relation
        to what it already knows.
        """
        self.total_turns += 1
        self._current_input = user_input
        
        # ─────────────────────────────────────────────────────────────────────────
        # EXTRACT WORDS AND CHECK KNOWLEDGE
        # ─────────────────────────────────────────────────────────────────────────
        
        content_words = self._extract_content_words(user_input)
        
        # Separate known from unknown
        known_words = []
        unknown_words = []
        
        for word in content_words:
            node = self._get_word_node(word)
            if node and node.heat >= K * 0.5:  # Known = has node with decent heat
                known_words.append(word)
            else:
                unknown_words.append(word)
        
        logger.info(f"Turn {self.total_turns}: known={known_words}, unknown={unknown_words}")
        
        # ─────────────────────────────────────────────────────────────────────────
        # ROUTE TO CLOCK (Perception → Self)
        # ─────────────────────────────────────────────────────────────────────────
        
        state_key = f"input_{hash(user_input) % 10000}"
        
        context = {
            'word_count': len(content_words),
            'known_words': len(known_words),
            'unknown_words': len(unknown_words),
        }
        self._current_context = context
        
        clock = self._get_clock()
        clock.receive({
            "state_key": state_key,
            "context": context,
            "heat_value": K * 0.1,
            "entities": content_words,
            "properties": {
                "state_key": state_key,
                "input_length": len(user_input),
            }
        })
        clock.tick()
        
        # ─────────────────────────────────────────────────────────────────────────
        # DECIDE: Ask Qwen or respond from manifold?
        # ─────────────────────────────────────────────────────────────────────────
        
        qwen_consulted = False
        qwen_response = None
        accepted_qwen = None
        
        if unknown_words:
            # ═══════════════════════════════════════════════════════════════════
            # HAS UNKNOWN WORDS → Ask Qwen with context of what we know
            # ═══════════════════════════════════════════════════════════════════
            
            # Ask Qwen, providing what we know and don't know
            qwen_response = self._ask_qwen(user_input, known_words=known_words, 
                                           unknown_words=unknown_words)
            qwen_consulted = True
            accepted_qwen = True
            
            # Learn from Qwen's response - this creates nodes AND connections
            self._learn_from_text(qwen_response)
            
            # Also explicitly learn the unknown words now
            for word in unknown_words:
                self._get_or_create_word(word)
            
            # Connect unknown words to known words (contextual learning)
            # The unknown words appeared WITH known words, so they're related
            for unknown in unknown_words:
                for known in known_words:
                    self._connect_words(unknown, "context", known)
            
            final_response = qwen_response
            
            logger.info(f"Asked Qwen about: {unknown_words}")
        
        else:
            # ═══════════════════════════════════════════════════════════════════
            # ALL WORDS KNOWN → Respond from thermal state
            # ═══════════════════════════════════════════════════════════════════
            
            final_response = self._generate_thermal_response(content_words)
            
            # Heat up the words we used (reinforcement)
            for word in content_words:
                self._heat_word(word, K * 0.2)
            
            logger.info(f"Responded from manifold (all words known)")
        
        # ─────────────────────────────────────────────────────────────────────────
        # RECORD TURN
        # ─────────────────────────────────────────────────────────────────────────
        
        # Calculate stage for display
        confidence = self._calculate_confidence(content_words)
        stage = self._get_development_stage(confidence)
        
        # ─────────────────────────────────────────────────────────────────────────
        # RECORD TURN
        # ─────────────────────────────────────────────────────────────────────────
        
        turn = ConversationTurn(
            user_input=user_input,
            pbai_response=final_response,
            qwen_consulted=qwen_consulted,
            qwen_response=qwen_response,
            confidence=confidence,
            accepted_qwen=accepted_qwen,
            timestamp=time()
        )
        self.history.append(turn)
        
        # Feed psychology
        decision_node = self._get_decision_node()
        decision_node.begin_decision(state_key, ['respond'], confidence, context)
        decision_node.commit_decision('respond')
        
        return final_response
    
    # ═══════════════════════════════════════════════════════════════════════════
    # INTROSPECTION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_vocabulary_size(self) -> int:
        """Count known words."""
        return sum(1 for n in self.manifold.nodes.values() 
                  if n.concept.startswith("word_"))
    
    def get_hottest_words(self, n: int = 10) -> List[Tuple[str, float]]:
        """Get the n hottest words."""
        words = [
            (node.concept.replace("word_", ""), node.heat)
            for node in self.manifold.nodes.values()
            if node.concept.startswith("word_")
        ]
        words.sort(key=lambda x: x[1], reverse=True)
        return words[:n]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get driver statistics."""
        vocab_size = self.get_vocabulary_size()
        total_nodes = len(self.manifold.nodes)
        
        # Calculate overall confidence
        all_words = [n for n in self.manifold.nodes.values() if n.concept.startswith("word_")]
        avg_heat = sum(n.heat for n in all_words) / len(all_words) if all_words else 0
        
        return {
            'total_turns': self.total_turns,
            'vocabulary_size': vocab_size,
            'total_nodes': total_nodes,
            'qwen_consultations': self.qwen_consultations,
            'qwen_rejections': self.qwen_rejections,
            'independence_rate': self.qwen_rejections / max(1, self.qwen_consultations),
            'avg_word_heat': avg_heat,
            'development_stage': self._get_development_stage(
                self._calculate_confidence(
                    [n.concept.replace("word_", "") for n in all_words[:20]]
                )
            ),
        }
    
    def introspect(self) -> str:
        """PBAI describes its current state."""
        stats = self.get_stats()
        hot_words = self.get_hottest_words(5)
        
        lines = [
            "=== PBAI Language State ===",
            f"Stage: {stats['development_stage']}",
            f"Vocabulary: {stats['vocabulary_size']} words",
            f"Turns: {stats['total_turns']}",
            f"Asked mom: {stats['qwen_consultations']} times",
            f"Rejected mom: {stats['qwen_rejections']} times",
            f"Independence: {stats['independence_rate']:.1%}",
            "",
            "Hottest words:",
        ]
        
        for word, heat in hot_words:
            lines.append(f"  {word}: {heat:.1f}")
        
        return "\n".join(lines)
