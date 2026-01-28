"""
PBAI Voice - Talk directly with PBAI

Unlike the relay chat, this interface makes PBAI's manifold state
drive the conversation. What PBAI "says" emerges from:
- What nodes are hot (what it's focused on)
- What connections exist (what it associates)
- What's curious (unexplored areas)
- What it's learned (hot decisions in tasks)

The LLM's role is ONLY to format PBAI's thermal state into natural language.
It doesn't decide what to say - PBAI does.

Usage:
    python -m gym_adapter.pbai_voice
    python -m gym_adapter.pbai_voice --no-llm  # Raw manifold output
"""

import sys
import os
import logging
from time import time
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core import Manifold, search, K
from core.nodes import Node
from core.node_constants import get_growth_path


def extract_concepts(text: str) -> list:
    """Extract concepts from text (simple word extraction)."""
    import re
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    stopwords = {'the', 'and', 'but', 'for', 'are', 'was', 'were', 'been', 'being',
                 'have', 'has', 'had', 'does', 'did', 'will', 'would', 'could', 'should',
                 'may', 'might', 'must', 'shall', 'can', 'need', 'dare', 'ought', 'used',
                 'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom', 'whose',
                 'where', 'when', 'why', 'how', 'all', 'each', 'every', 'both', 'few',
                 'more', 'most', 'other', 'some', 'such', 'only', 'own', 'same', 'than',
                 'too', 'very', 'just', 'also', 'now', 'here', 'there', 'then', 'once',
                 'with', 'from', 'into', 'about', 'your', 'you', 'they', 'them', 'their'}
    return [w for w in words if w not in stopwords]


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)

# Default growth path - ONE brain for all tasks
VOICE_GROWTH_PATH = get_growth_path("growth_map.json")


@dataclass
class PBAIMind:
    """
    Represents PBAI's current mental state extracted from manifold.
    
    This is what PBAI "thinks" at any moment - derived entirely from
    the thermal manifold, not from any LLM.
    """
    # Attention - what's hot right now
    focus: List[Tuple[str, float]]  # (concept, heat) pairs
    
    # Memory - what it knows about
    knowledge_areas: List[str]
    
    # Curiosity - what it wants to explore
    curious_about: List[str]
    
    # Confidence - how sure it is about things (0-1)
    # Based on: experience depth, decision clarity, consistency
    confidence: float
    
    # Mood - derived from recent performance and state
    # "learning" - actively acquiring new knowledge
    # "confident" - strong preferences, good performance
    # "curious" - exploring new territory
    # "uncertain" - low experience or inconsistent results
    # "focused" - deep in a specific task
    mood: str
    
    # Active tasks
    active_tasks: List[str]
    
    # Recent learnings
    recent_decisions: List[Tuple[str, str, float]]  # (task, decision, heat)
    
    # NEW: Experience metrics
    total_experience: int  # Total traversal counts across all axes
    decision_clarity: float  # 0-1, how distinct are good vs bad choices
    specialization: str  # Which task has most heat
    
    # NEW: Performance indicators  
    hot_decisions: int  # Decisions with heat > 2*K (reinforced positively)
    cold_decisions: int  # Decisions with heat < 0.5*K (reinforced negatively)
    neutral_decisions: int  # Decisions near K (unexplored)


class PBAIVoice:
    """
    PBAI's voice - converts manifold state to communication.
    
    This is NOT an LLM making things up. This is PBAI expressing
    its actual thermal state in words.
    
    Persistence:
        - Automatically loads existing growth map on startup
        - Saves after each input processing
        - Use /save, /load, /reset commands for manual control
    """
    
    def __init__(self, manifold: Optional[Manifold] = None, use_llm: bool = True,
                 growth_path: str = None, auto_persist: bool = True):
        """
        Initialize PBAI voice interface.
        
        Args:
            manifold: Existing manifold to use (or None to get the ONE PBAI manifold)
            use_llm: Whether to use LLM for response formatting
            growth_path: Path to save/load growth map (default: growth/growth_map.json)
            auto_persist: Whether to auto-save after each input
        """
        from core import get_pbai_manifold
        
        self.growth_path = growth_path if growth_path else VOICE_GROWTH_PATH
        self.auto_persist = auto_persist
        
        if manifold:
            # Use provided manifold (driver pattern)
            self.manifold = manifold
        else:
            # Get the ONE PBAI manifold
            self.manifold = get_pbai_manifold(self.growth_path)
        
        logger.info(f"Voice interface ready: {len(self.manifold.nodes)} nodes")
        
        self.use_llm = use_llm
        self.llm = None
        
        if use_llm:
            self._init_llm()
        
        # Conversation history for context
        self.history: List[Tuple[str, str]] = []  # (user, pbai) pairs
    
    def _try_load(self) -> bool:
        """Legacy method - kept for compatibility."""
        import os
        return os.path.exists(self.growth_path)
    
    def save(self) -> str:
        """Save current growth map."""
        import os
        os.makedirs(os.path.dirname(self.growth_path) or ".", exist_ok=True)
        self.manifold.save_growth_map(self.growth_path)
        return f"Saved to {self.growth_path}"
    
    def reset(self) -> str:
        """Reset manifold to fresh state (for testing only)."""
        from core import get_pbai_manifold, reset_pbai_manifold
        import os
        
        if os.path.exists(self.growth_path):
            os.remove(self.growth_path)
        reset_pbai_manifold()
        self.manifold = get_pbai_manifold(self.growth_path)
        self.history = []
        return "Reset complete. Fresh start."
    
    def _init_llm(self):
        """Initialize LLM for formatting (not thinking)."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            model_name = "Qwen/Qwen2.5-3B-Instruct"
            logger.info(f"Loading formatter: {model_name}...")
            
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
            logger.info("Formatter ready")
            
        except Exception as e:
            logger.warning(f"LLM not available: {e}")
            self.use_llm = False
    
    def read_mind(self) -> PBAIMind:
        """
        Read PBAI's current mental state from the manifold.
        
        This is pure introspection - no LLM involved.
        Metrics are derived from actual manifold state.
        """
        nodes = list(self.manifold.nodes.values())
        
        # Focus: top 5 hottest non-bootstrap nodes
        hot_nodes = sorted(
            [n for n in nodes if not n.concept.startswith('bootstrap')],
            key=lambda n: n.heat,
            reverse=True
        )[:5]
        focus = [(n.concept, n.heat) for n in hot_nodes]
        
        # Knowledge areas: unique prefixes
        areas = set()
        for n in nodes:
            if '_' in n.concept:
                prefix = n.concept.split('_')[0]
                if not prefix.startswith('bootstrap'):
                    areas.add(prefix)
        knowledge_areas = list(areas)
        
        # Curiosity: nodes with exactly K heat (unexplored but noted)
        curious = [n.concept for n in nodes 
                   if abs(n.heat - K) < 0.1 and not n.concept.startswith('bootstrap')][:5]
        
        # ═══════════════════════════════════════════════════════════════════
        # EXPERIENCE METRICS (from axes)
        # ═══════════════════════════════════════════════════════════════════
        
        total_experience = 0
        hot_decisions = 0
        cold_decisions = 0
        neutral_decisions = 0
        
        # Count traversals across all semantic axes
        if self.manifold.self_node and self.manifold.self_node.frame:
            for axis in self.manifold.self_node.frame.semantic_axes.values():
                total_experience += axis.traversal_count
        
        for n in nodes:
            if n.frame and n.frame.semantic_axes:
                for axis in n.frame.semantic_axes.values():
                    total_experience += axis.traversal_count
            
            # Classify by heat (decisions are nodes with underscores)
            if '_' in n.concept and not n.concept.startswith('bootstrap'):
                if n.heat > 2 * K:
                    hot_decisions += 1
                elif n.heat < 0.5 * K:
                    cold_decisions += 1
                else:
                    neutral_decisions += 1
        
        # ═══════════════════════════════════════════════════════════════════
        # DECISION CLARITY (heat spread)
        # ═══════════════════════════════════════════════════════════════════
        
        decision_heats = [n.heat for n in nodes 
                         if '_' in n.concept and not n.concept.startswith('bootstrap')]
        if decision_heats and len(decision_heats) > 1:
            max_h = max(decision_heats)
            min_h = min(decision_heats)
            # Clarity = spread normalized by expected range (0 to 5K)
            decision_clarity = min(1.0, (max_h - min_h) / (5 * K))
        else:
            decision_clarity = 0.0
        
        # ═══════════════════════════════════════════════════════════════════
        # SPECIALIZATION (which task has most heat)
        # ═══════════════════════════════════════════════════════════════════
        
        task_heat = {}
        for n in nodes:
            # Skip bootstrap nodes entirely
            if n.concept.startswith('bootstrap'):
                continue
            
            if n.righteousness == 1.0:
                task_heat[n.concept] = n.heat
            elif '_' in n.concept:
                prefix = n.concept.split('_')[0]
                if prefix not in task_heat:
                    task_heat[prefix] = 0
                task_heat[prefix] += n.heat
        
        if task_heat:
            specialization = max(task_heat.keys(), key=lambda k: task_heat[k])
        else:
            specialization = "none"
        
        # ═══════════════════════════════════════════════════════════════════
        # CONFIDENCE AND MOOD FROM MANIFOLD PSYCHOLOGY
        # Derived from Identity/Conscience/Ego - NO PYTHON CHEATS
        # ═══════════════════════════════════════════════════════════════════
        
        # Get confidence from Ego's heat and experience
        confidence = self.manifold.get_confidence()
        
        # Get mood from psychology node heat ratios
        mood = self.manifold.get_mood()
        
        # Active tasks: righteous frames
        tasks = [n.concept for n in nodes 
                 if n.righteousness == 1.0 and not n.concept.startswith('bootstrap')]
        
        # Recent decisions: task decisions with heat != K
        decisions = []
        for n in nodes:
            if '_' in n.concept and n.heat != K and not n.concept.startswith('bootstrap'):
                parts = n.concept.rsplit('_', 1)
                if len(parts) == 2:
                    task = parts[0]
                    decision = parts[1]
                    decisions.append((task, decision, n.heat))
        decisions.sort(key=lambda x: abs(x[2] - K), reverse=True)
        
        return PBAIMind(
            focus=focus,
            knowledge_areas=knowledge_areas,
            curious_about=curious,
            confidence=confidence,
            mood=mood,
            active_tasks=tasks,
            recent_decisions=decisions[:5],
            total_experience=total_experience,
            decision_clarity=decision_clarity,
            specialization=specialization,
            hot_decisions=hot_decisions,
            cold_decisions=cold_decisions,
            neutral_decisions=neutral_decisions
        )
    
    def process_input(self, user_input: str) -> str:
        """
        Process user input and generate PBAI's response.
        
        Flow:
        1. Extract concepts from input
        2. Detect relationships ("X is Y") and build connections
        3. Detect questions and resolve from connections
        4. Query manifold for each concept
        5. Read resulting mind state
        6. Generate response from state
        """
        # Extract concepts
        concepts = extract_concepts(user_input)
        lower_input = user_input.lower()
        
        # Detect and build relationships
        relationships = self._extract_relationships(user_input)
        for subject, predicate, obj, direction in relationships:
            self._build_relationship(subject, predicate, obj, direction)
        
        # Process concepts into manifold (creates nodes if needed)
        for concept in concepts:
            search(concept, self.manifold)
        
        # Check if this is a question we can answer from relationships
        answer = self._try_answer_question(user_input, concepts)
        if answer:
            self.history.append((user_input, answer))
            return answer
        
        # Read current mind state
        mind = self.read_mind()
        
        # Generate response
        if self.use_llm and self.llm:
            response = self._format_with_llm(user_input, concepts, mind)
        else:
            response = self._format_raw(user_input, concepts, mind)
        
        # Track history
        self.history.append((user_input, response))
        
        # Auto-save if enabled
        if self.auto_persist:
            try:
                self.save()
            except Exception as e:
                logger.warning(f"Auto-save failed: {e}")
        
        return response
    
    def _extract_relationships(self, text: str) -> list:
        """
        Extract relationships from text.
        
        Returns list of (subject, predicate, object, direction) tuples.
        
        Direction semantics:
        - 'n' = positive assertion (is, has, equals)
        - 's' = negative assertion (is not, doesn't)
        - 'u' = generalization (is a type of, category)
        - 'd' = specification (is an example of)
        - 'e'/'w' = association (related to, with)
        """
        import re
        relationships = []
        lower = text.lower().strip()
        
        # Skip extraction if this is a question
        is_question = lower.endswith('?') or lower.startswith(('what', 'who', 'where', 'when', 'why', 'how', 'do ', 'does ', 'is ', 'are ', 'can ', 'could ', 'would ', 'will '))
        
        if is_question:
            return relationships  # Don't extract relationships from questions
        
        # "Your X is Y" - property of self (but not "is your" which is different)
        # Match: "Your name is PBAI" but not "What is your name"
        for match in re.finditer(r"(?:^|[.!]\s*)your\s+(\w+)\s+is\s+(\w+)", lower):
            prop, value = match.groups()
            if prop not in ('favorite',):  # Skip compound phrases like "your favorite color"
                relationships.append(("self", prop, value, 'n'))
        
        # "My X is Y" - property of user
        for match in re.finditer(r"(?:^|[.!]\s*)my\s+(\w+)\s+is\s+(\w+)", lower):
            prop, value = match.groups()
            relationships.append(("user", prop, value, 'n'))
        
        # "You are X" - identity of self
        # Must be at start of sentence or after punctuation, and X must be a name-like word
        for match in re.finditer(r"(?:^|[.!]\s*)you\s+are\s+(\w+)", lower):
            value = match.group(1)
            # Filter out common non-identity words
            skip_words = ('a', 'an', 'the', 'not', 'being', 'going', 'doing', 
                         'identified', 'called', 'named', 'known', 'able', 'welcome',
                         'right', 'wrong', 'correct', 'sure', 'here', 'there')
            if value not in skip_words and len(value) > 1:
                relationships.append(("self", "identity", value, 'n'))
                relationships.append(("self", "name", value, 'n'))
        
        # "I am X" - identity of user (must be at start)
        for match in re.finditer(r"(?:^|[.!]\s*)i\s+am\s+(\w+)", lower):
            value = match.group(1)
            skip_words = ('a', 'an', 'the', 'not', 'going', 'doing', 'sure', 'here', 'there')
            if value not in skip_words and len(value) > 1:
                relationships.append(("user", "identity", value, 'n'))
                relationships.append(("user", "name", value, 'n'))
        
        # "X is your Y" - X has role Y for self (but not in questions)
        for match in re.finditer(r"(\w+)\s+is\s+your\s+(\w+)", lower):
            value, role = match.groups()
            skip_subjects = ('it', 'this', 'that', 'what', 'who', 'which', 'here', 'there')
            if value not in skip_subjects:
                relationships.append(("self", role, value, 'n'))
        
        # "I created you" - user created self
        if re.search(r"i\s+created\s+you", lower):
            relationships.append(("self", "creator", "user", 'n'))
            relationships.append(("user", "created", "self", 'n'))
        
        # "X created Y" (not "I created you")
        for match in re.finditer(r"(\w+)\s+created\s+(\w+)", lower):
            creator, created = match.groups()
            if creator != "i" and created != "you":
                relationships.append((created, "creator", creator, 'n'))
        
        # "X is a/an Y" - type/category relationship (up direction)
        for match in re.finditer(r"(\w+)\s+is\s+(?:a|an)\s+(\w+)", lower):
            instance, category = match.groups()
            skip_subjects = ('it', 'this', 'that', 'what', 'who', 'which', 'there')
            if instance not in skip_subjects:
                relationships.append((instance, "type", category, 'u'))
        
        # "X is not Y" - negative assertion
        for match in re.finditer(r"(\w+)\s+(?:is\s+not|isn't|am\s+not)\s+(\w+)", lower):
            subj, obj = match.groups()
            if subj == 'i':
                subj = 'user'
            if obj not in ('a', 'an', 'the'):
                relationships.append((subj, "not", obj, 's'))
        
        return relationships
    
    def _build_relationship(self, subject: str, predicate: str, obj: str, direction: str):
        """
        Build a conception (righteousness connection) in the manifold.
        
        Creates nodes if needed, creates conceptions with polarity.
        Repeated calls strengthen the conception (traversal_count increases).
        
        Direction maps to polarity:
            'n' = positive assertion (+1)
            's' = negative assertion (-1)
            'u', 'd', 'e', 'w' = positive by default (+1)
        """
        # Handle special subjects
        if subject == "self":
            subj_node = self.manifold.self_node
        elif subject == "user":
            # Create/get user node
            subj_node = self.manifold.get_node_by_concept("user")
            if not subj_node:
                subj_node = self._ensure_node("user")
        else:
            subj_node = self._ensure_node(subject)
        
        # Handle special objects
        if obj == "self":
            obj_node = self.manifold.self_node
        elif obj == "user":
            obj_node = self.manifold.get_node_by_concept("user")
            if not obj_node:
                obj_node = self._ensure_node("user")
        else:
            obj_node = self._ensure_node(obj)
        
        if not subj_node or not obj_node:
            return
        
        # Map direction to polarity
        polarity = -1 if direction == 's' else +1
        
        # Create or strengthen conception
        conception = subj_node.conceive(predicate, obj_node.id, polarity)
        
        if conception.traversal_count > 1:
            logger.info(f"Strengthened: {subj_node.concept} --{predicate}[{'+' if polarity > 0 else '-'}]--> {obj_node.concept} (count={conception.traversal_count})")
        else:
            logger.info(f"Conceived: {subj_node.concept} --{predicate}[{'+' if polarity > 0 else '-'}]--> {obj_node.concept}")
        
        # Add heat to both nodes (they were mentioned)
        if subj_node.concept != "self":
            subj_node.add_heat(K * 0.5)
        obj_node.add_heat(K * 0.5)
    
    def _ensure_node(self, concept: str) -> Node:
        """Get existing node or create new one for concept."""
        node = self.manifold.get_node_by_concept(concept)
        if not node:
            result = search(concept, self.manifold)
            node = result.center
        return node
    
    def _try_answer_question(self, user_input: str, concepts: List[str]) -> Optional[str]:
        """
        Try to answer a question by following connections.
        
        Questions like "What is your name?" should follow:
        self --name--> ? and return the target.
        
        Multi-hop questions like "What is the name of your creator?" should follow:
        self --creator--> X --name--> ? (Ego's inference engine)
        """
        import re
        lower = user_input.lower()
        
        # ═══════════════════════════════════════════════════════════════════
        # MULTI-HOP INFERENCE (Ego's job)
        # "What is the X of the Y of Z?" → traverse_chain
        # ═══════════════════════════════════════════════════════════════════
        
        # Pattern: "What is the X of your Y?" or "What is the X of the Y?"
        # e.g. "What is the name of your creator?" → ["creator", "name"]
        match = re.search(r"what(?:'s|\s+is)\s+the\s+(\w+)\s+of\s+(?:your\s+)?(\w+)", lower)
        if match:
            prop2, prop1 = match.groups()  # name, creator → [creator, name]
            result = self.manifold.infer("self", [prop1, prop2])
            if result:
                return f"The {prop2} of my {prop1} is {result}."
        
        # Pattern: "What is the X of the person/one who Y you?"
        # e.g. "What is the name of the person who created you?" → ["creator", "name"]
        match = re.search(r"what(?:'s|\s+is)\s+the\s+(\w+)\s+of\s+(?:the\s+)?(?:\w+\s+)?who\s+(\w+)\s+you", lower)
        if match:
            prop, verb = match.groups()  # name, created
            # Map verb to predicate
            verb_to_pred = {
                "created": "creator",
                "made": "creator",
                "built": "creator",
            }
            pred = verb_to_pred.get(verb, verb)
            result = self.manifold.infer("self", [pred, prop])
            if result:
                return f"The {prop} of my {pred} is {result}."
        
        # Pattern: "What is my Y's X?" (e.g., "What is my creator's name?")
        match = re.search(r"what(?:'s|\s+is)\s+(?:my|your)\s+(\w+)(?:'s|s)\s+(\w+)", lower)
        if match:
            prop1, prop2 = match.groups()  # creator, name
            subject = "self" if "your" in lower else "user"
            result = self.manifold.infer(subject, [prop1, prop2])
            if result:
                return f"My {prop1}'s {prop2} is {result}." if subject == "self" else f"Your {prop1}'s {prop2} is {result}."
        
        # ═══════════════════════════════════════════════════════════════════
        # SINGLE-HOP QUERIES (existing logic)
        # ═══════════════════════════════════════════════════════════════════
        
        # Pattern: "What is your X?" or "What's your X?"
        match = re.search(r"what(?:'s|\s+is)\s+your\s+(\w+)", lower)
        if match:
            prop = match.group(1)
            return self._query_property("self", prop)
        
        # Pattern: "Who is your X?" (creator, friend, etc.)
        match = re.search(r"who\s+is\s+your\s+(\w+)", lower)
        if match:
            prop = match.group(1)
            return self._query_property("self", prop)
        
        # Pattern: "Who are you?" or "What are you?"
        if re.search(r"(?:who|what)\s+are\s+you", lower):
            return self._query_property("self", "identity") or self._query_property("self", "name")
        
        # Pattern: "Do you have a/an X?" or "Do you have X?"
        match = re.search(r"do\s+you\s+have\s+(?:an\s+|a\s+)?(\w+)", lower)
        if match:
            prop = match.group(1)
            return self._query_existence("self", prop)
        
        # Pattern: "What is my X?"
        match = re.search(r"what(?:'s|\s+is)\s+my\s+(\w+)", lower)
        if match:
            prop = match.group(1)
            return self._query_property("user", prop)
        
        # Pattern: "Who am I?"
        if re.search(r"who\s+am\s+i", lower):
            return self._query_property("user", "identity") or self._query_property("user", "name")
        
        # Pattern: "Who created you?"
        if re.search(r"who\s+created\s+you", lower):
            return self._query_property("self", "creator")
        
        return None
    
    def _query_existence(self, subject: str, prop: str) -> Optional[str]:
        """
        Query whether a subject HAS a property (existence check).
        
        "Do you have an identity?" → Check if self has any identity connection.
        """
        # Get subject node
        if subject == "self":
            subj_node = self.manifold.self_node
        else:
            subj_node = self.manifold.get_node_by_concept(subject)
        
        if not subj_node:
            return None
        
        # Look for conception with this predicate
        conception = subj_node.get_conception(prop)
        
        if conception:
            target = self.manifold.get_node(conception.target_id)
            if target:
                count = conception.traversal_count
                confidence = "Yes," if count >= 2 else "I think so."
                if subject == "self":
                    return f"{confidence} My {prop} is {target.concept}."
                else:
                    return f"{confidence} Your {prop} is {target.concept}."
        
        # No conception found
        if subject == "self":
            return f"I don't know if I have a {prop} yet. You could tell me."
        
        return None
    
    def _query_property(self, subject: str, prop: str) -> Optional[str]:
        """
        Query a property of a subject by following conceptions.
        
        Returns natural language answer if found, None otherwise.
        """
        # Get subject node
        if subject == "self":
            subj_node = self.manifold.self_node
        else:
            subj_node = self.manifold.get_node_by_concept(subject)
        
        if not subj_node:
            return None
        
        # Look for conception with this predicate
        conception = subj_node.get_conception(prop)
        
        if conception:
            target = self.manifold.get_node(conception.target_id)
            if target:
                count = conception.traversal_count
                is_verified = conception.is_verified  # count >= 3
                
                if subject == "self":
                    if prop in ("name", "identity"):
                        return f"My {prop} is {target.concept}." if count >= 2 else f"I think my {prop} is {target.concept}."
                    elif prop == "creator":
                        if target.concept == "user":
                            return f"You created me." if count >= 2 else f"I believe you created me."
                        return f"{target.concept.capitalize()} created me." if count >= 2 else f"I think {target.concept} created me."
                    else:
                        return f"My {prop} is {target.concept}."
                elif subject == "user":
                    if prop in ("name", "identity"):
                        return f"Your {prop} is {target.concept}."
                    else:
                        return f"Your {prop} is {target.concept}."
        
        # No conception found - express uncertainty
        if subject == "self":
            return f"I don't know my {prop} yet."
        
        return None
    
    def _format_raw(self, user_input: str, concepts: List[str], mind: PBAIMind) -> str:
        """
        Generate response directly from mind state - no LLM.
        
        This is PBAI speaking as itself, unfiltered.
        """
        lines = []
        
        # Acknowledge what was heard
        if concepts:
            lines.append(f"[Heard: {', '.join(concepts[:5])}]")
        
        # Show any relationships detected
        relationships = self._extract_relationships(user_input)
        if relationships:
            rel_strs = [f"{s}--{p}-->{o}" for s, p, o, d in relationships[:3]]
            lines.append(f"[Relationships: {', '.join(rel_strs)}]")
        
        # Express current focus
        if mind.focus:
            top = mind.focus[0]
            lines.append(f"[Focus: {top[0]} (heat={top[1]:.1f})]")
        
        # Express mood
        lines.append(f"[State: {mind.mood}]")
        
        # Show conceptions from Self
        self_concs = []
        for predicate, conc in self.manifold.self_node.frame.semantic_axes.items():
            target = self.manifold.get_node(conc.target_id)
            if target:
                pol = "+" if conc.polarity > 0 else "-"
                self_concs.append(f"{predicate}[{pol}]→{target.concept}(×{conc.traversal_count})")
        if self_concs:
            lines.append(f"[Self knows: {', '.join(self_concs[:5])}]")
        
        # Share relevant knowledge
        relevant = [f for f, h in mind.focus if any(c in f.lower() for c in [c.lower() for c in concepts])]
        if relevant:
            lines.append(f"[Related: {', '.join(relevant[:3])}]")
        
        # Express curiosity
        if mind.curious_about:
            lines.append(f"[Curious: {', '.join(mind.curious_about[:2])}]")
        
        # Share recent learnings if relevant
        if mind.recent_decisions:
            top_decision = mind.recent_decisions[0]
            if top_decision[2] > K:
                lines.append(f"[Learned: {top_decision[0]}_{top_decision[1]} works (heat={top_decision[2]:.1f})]")
            elif top_decision[2] < K:
                lines.append(f"[Learned: {top_decision[0]}_{top_decision[1]} doesn't work (heat={top_decision[2]:.1f})]")
        
        return "\n".join(lines)
    
    def _format_with_llm(self, user_input: str, concepts: List[str], mind: PBAIMind) -> str:
        """
        Use LLM to format mind state into natural language.
        
        IMPORTANT: The LLM doesn't decide what to say. It formats
        what PBAI already "thinks" based on manifold state.
        """
        # Gather PBAI's known facts from conceptions
        known_facts = []
        for predicate, conc in self.manifold.self_node.frame.semantic_axes.items():
            target = self.manifold.get_node(conc.target_id)
            if target:
                if predicate == "name":
                    known_facts.append(f"My name is {target.concept} (confidence: {conc.traversal_count})")
                elif predicate == "identity":
                    known_facts.append(f"My identity is {target.concept} (confidence: {conc.traversal_count})")
                elif predicate == "creator":
                    if target.concept == "user":
                        known_facts.append(f"You (the user) created me (confidence: {conc.traversal_count})")
                    else:
                        known_facts.append(f"{target.concept} created me (confidence: {conc.traversal_count})")
                else:
                    pol = "is" if conc.polarity > 0 else "is not"
                    known_facts.append(f"My {predicate} {pol} {target.concept} (confidence: {conc.traversal_count})")
        
        # Gather known facts about user
        user_node = self.manifold.get_node_by_concept("user")
        if user_node:
            for predicate, conc in user_node.frame.semantic_axes.items():
                target = self.manifold.get_node(conc.target_id)
                if target:
                    if predicate == "name":
                        known_facts.append(f"User's name is {target.concept}")
                    else:
                        known_facts.append(f"User's {predicate} is {target.concept}")
        
        facts_str = "\n".join(f"  - {f}" for f in known_facts) if known_facts else "  - None yet"
        
        # Build prompt that constrains LLM to just format
        prompt = f"""You are formatting PBAI's response. PBAI is an AI with a thermal manifold brain.
DO NOT make things up. Only express what PBAI actually knows from its current state.
NEVER contradict the known facts below - these are things PBAI definitely knows.

PBAI's Known Facts (DO NOT CONTRADICT):
{facts_str}

PBAI's Current State:
- Mood: {mind.mood}
- Confidence: {mind.confidence:.0%}
- Focus: {', '.join(f'{f}({h:.1f})' for f, h in mind.focus[:3]) if mind.focus else 'nothing specific'}
- Knowledge areas: {', '.join(mind.knowledge_areas[:5]) if mind.knowledge_areas else 'none yet'}
- Curious about: {', '.join(mind.curious_about[:3]) if mind.curious_about else 'nothing specific'}

User said: "{user_input}"
Concepts heard: {concepts}

Write PBAI's response. Keep it brief (1-3 sentences). 
If asked about something PBAI knows (from Known Facts), answer confidently.
If asked about something PBAI doesn't know, say so honestly and express curiosity.
Don't use phrases like "As PBAI" or "I am PBAI". Just respond naturally as PBAI.

PBAI:"""
        
        try:
            response = self.llm(prompt)
            # Clean up response
            response = response.strip()
            # Remove any meta-commentary the LLM added
            for prefix in ["As PBAI,", "PBAI:", "I think", "I believe"]:
                if response.startswith(prefix):
                    response = response[len(prefix):].strip()
            return response
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._format_raw(user_input, concepts, mind)
    
    def introspect(self) -> str:
        """
        PBAI describes its own state.
        
        Useful for debugging and understanding what PBAI "knows".
        """
        mind = self.read_mind()
        
        lines = [
            "=== PBAI Introspection ===",
            f"Mood: {mind.mood}",
            f"Confidence: {mind.confidence:.0%}",
            "",
            "Focus (hottest):",
        ]
        
        for concept, heat in mind.focus:
            bar = "█" * min(10, int(heat))
            lines.append(f"  {concept}: {heat:.1f} {bar}")
        
        if mind.knowledge_areas:
            lines.append("")
            lines.append(f"Knowledge: {', '.join(mind.knowledge_areas)}")
        
        if mind.curious_about:
            lines.append("")
            lines.append(f"Curious about: {', '.join(mind.curious_about)}")
        
        if mind.active_tasks:
            lines.append("")
            lines.append(f"Tasks: {', '.join(mind.active_tasks)}")
        
        if mind.recent_decisions:
            lines.append("")
            lines.append("Recent learnings:")
            for task, decision, heat in mind.recent_decisions[:5]:
                status = "✓" if heat > K else "✗" if heat < K else "?"
                lines.append(f"  {status} {task}_{decision}: {heat:.2f}")
        
        lines.append("")
        lines.append(f"Total nodes: {len(self.manifold.nodes)}")
        
        return "\n".join(lines)
    
    def share_knowledge(self, topic: str) -> str:
        """
        PBAI shares what it knows about a topic.
        """
        # Find nodes related to topic
        related = []
        for node in self.manifold.nodes.values():
            if topic.lower() in node.concept.lower():
                related.append((node.concept, node.heat))
        
        if not related:
            return f"I don't have any thermal memory about '{topic}' yet."
        
        related.sort(key=lambda x: x[1], reverse=True)
        
        lines = [f"What I know about '{topic}':"]
        for prop, heat in related[:10]:
            status = "hot" if heat > K else "cold" if heat < K else "neutral"
            lines.append(f"  {prop}: {status} ({heat:.1f})")
        
        return "\n".join(lines)
    
    def show_connections(self) -> str:
        """
        Show all conceptions in the manifold with capability levels.
        
        Capability levels: R=righteous, O=ordered, M=movable, G=graphic
        """
        lines = ["=== Conception Graph ===", ""]
        
        # Self's conceptions first
        lines.append("SELF (conceptions):")
        if self.manifold.self_node.frame.semantic_axes:
            for predicate, conc in sorted(self.manifold.self_node.frame.semantic_axes.items()):
                target = self.manifold.get_node(conc.target_id)
                target_name = target.concept if target else "???"
                strength = "█" * min(5, conc.traversal_count)
                pol = "+" if conc.polarity > 0 else "-"
                cap = conc.capability[0].upper()  # R/O/M/G
                verified = "✓" if conc.is_verified else ""
                lines.append(f"  --{predicate}[{pol}]--> {target_name} [{strength}] (×{conc.traversal_count}) [{cap}] {verified}")
                
                # Show order elements if ordered
                if conc.is_ordered and conc.order.elements:
                    elem_names = []
                    for elem in conc.order.elements[:5]:  # Limit display
                        node = self.manifold.get_node(elem.node_id)
                        elem_names.append(node.concept if node else "???")
                    more = f"...+{len(conc.order.elements)-5}" if len(conc.order.elements) > 5 else ""
                    lines.append(f"      elements: [{', '.join(elem_names)}{more}]")
        else:
            lines.append("  (none)")
        
        lines.append("")
        
        # Other nodes with conceptions
        for node in sorted(self.manifold.nodes.values(), key=lambda n: -n.heat):
            if node.frame.semantic_axes:
                lines.append(f"{node.concept.upper()} (heat={node.heat:.1f}, R={node.righteousness:.2f}):")
                for predicate, conc in sorted(node.frame.semantic_axes.items()):
                    target = self.manifold.get_node(conc.target_id)
                    target_name = target.concept if target else "???"
                    strength = "█" * min(5, conc.traversal_count)
                    pol = "+" if conc.polarity > 0 else "-"
                    cap = conc.capability[0].upper()
                    lines.append(f"  --{predicate}[{pol}]--> {target_name} [{strength}] (×{conc.traversal_count}) [{cap}]")
        
        lines.append("")
        lines.append(f"Total nodes: {len(self.manifold.nodes)}")
        
        # Count conceptions by capability
        cap_counts = {"R": 0, "O": 0, "M": 0, "G": 0}
        for conc in self.manifold.self_node.frame.semantic_axes.values():
            cap_counts[conc.capability[0].upper()] += 1
        for node in self.manifold.nodes.values():
            for conc in node.frame.semantic_axes.values():
                cap_counts[conc.capability[0].upper()] += 1
        
        total = sum(cap_counts.values())
        lines.append(f"Conceptions: {total} (R:{cap_counts['R']} O:{cap_counts['O']} M:{cap_counts['M']} G:{cap_counts['G']})")
        
        return "\n".join(lines)


def run_voice_terminal():
    """Run PBAI voice in terminal mode."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Talk to PBAI")
    parser.add_argument("--no-llm", action="store_true", help="Disable LLM formatting")
    parser.add_argument("--growth-path", type=str, default=None,
                       help="Path to growth map file (default: growth/growth_map.json)")
    parser.add_argument("--no-persist", action="store_true", help="Disable auto-persistence")
    parser.add_argument("--fresh", action="store_true", help="Start fresh (ignore existing growth map)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PBAI Voice - Talk directly to PBAI")
    print("=" * 60)
    print()
    
    # Resolve growth path (use default if not specified)
    growth_path = args.growth_path if args.growth_path else VOICE_GROWTH_PATH
    
    # Handle fresh start
    if args.fresh and os.path.exists(growth_path):
        os.remove(growth_path)
        print("Starting fresh (deleted existing growth map)")
    
    voice = PBAIVoice(
        use_llm=not args.no_llm,
        growth_path=growth_path,
        auto_persist=not args.no_persist
    )
    
    # Show status
    loaded = os.path.exists(growth_path)
    conceptions = len(voice.manifold.self_node.frame.semantic_axes)
    
    print(f"Manifold: {len(voice.manifold.nodes)} nodes")
    print(f"Self conceptions: {conceptions}")
    print(f"LLM: {'ON' if voice.use_llm else 'OFF'}")
    print(f"Persistence: {'ON' if voice.auto_persist else 'OFF'} ({growth_path})")
    if loaded:
        print(f"  → Loaded existing knowledge!")
    print()
    print("Commands:")
    print("  /introspect   - PBAI describes its state")
    print("  /connections  - Show conception graph")
    print("  /know <topic> - What PBAI knows about topic")
    print("  /llm          - Toggle LLM formatting")
    print("  /save         - Save growth map now")
    print("  /reset        - Start fresh (delete all knowledge)")
    print("  /quit         - Exit")
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
            parts = user_input[1:].split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1] if len(parts) > 1 else ""
            
            if cmd == "quit":
                if voice.auto_persist:
                    voice.save()
                    print("Saved.")
                print("Goodbye!")
                break
            
            elif cmd == "introspect":
                print()
                print(voice.introspect())
                print()
            
            elif cmd == "connections":
                print()
                print(voice.show_connections())
                print()
            
            elif cmd == "know":
                if arg:
                    print()
                    print(voice.share_knowledge(arg))
                    print()
                else:
                    print("Usage: /know <topic>")
            
            elif cmd == "llm":
                voice.use_llm = not voice.use_llm
                print(f"LLM formatting: {'ON' if voice.use_llm else 'OFF'}")
            
            elif cmd == "save":
                result = voice.save()
                print(result)
            
            elif cmd == "reset":
                confirm = input("Are you sure? This deletes all learned knowledge. (yes/no): ")
                if confirm.lower() == "yes":
                    result = voice.reset()
                    print(result)
                else:
                    print("Reset cancelled.")
            
            else:
                print(f"Unknown command: {cmd}")
        
        else:
            response = voice.process_input(user_input)
            print(f"PBAI: {response}")
            print()


if __name__ == "__main__":
    run_voice_terminal()
