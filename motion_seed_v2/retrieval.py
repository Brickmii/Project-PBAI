"""
Motion Calendar Seed - Retrieval Module
Associative memory and cued recall.
"""

import time
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
from dataclasses import dataclass, field
import math

from core import Particle


@dataclass
class Association:
    """A link between two items with strength."""
    source: str  # domain or particle id
    target: str  # domain or particle id
    strength: float
    co_occurrences: int = 1
    last_activated: float = field(default_factory=time.time)
    
    def reinforce(self, amount: float = 0.1):
        """Strengthen association through use."""
        self.strength = min(1.0, self.strength + amount)
        self.co_occurrences += 1
        self.last_activated = time.time()
    
    def decay(self, rate: float = 0.01):
        """Weaken association over time."""
        elapsed = time.time() - self.last_activated
        decay_amount = rate * (elapsed / 3600)  # per hour
        self.strength = max(0.0, self.strength - decay_amount)


class AssociativeMemory:
    """
    Associative memory system.
    
    Tracks co-occurrence of items and builds retrievable associations.
    """
    
    def __init__(self):
        # Associations: source -> {target -> Association}
        self.associations: Dict[str, Dict[str, Association]] = defaultdict(dict)
        
        # Reverse index: target -> [sources]
        self.reverse_index: Dict[str, List[str]] = defaultdict(list)
        
        # Content index: raw_value -> [particle_ids]
        self.content_index: Dict[str, List[str]] = defaultdict(list)
        
        # Domain content: domain_id -> [raw_values]
        self.domain_content: Dict[str, List[Any]] = defaultdict(list)
        
        # Recent context window for co-occurrence
        self.context_window: List[Tuple[str, str, Any]] = []  # (domain_id, particle_id, raw_value)
        self.window_size = 20
        
        # Statistics
        self.total_associations = 0
        self.total_retrievals = 0
    
    def observe(self, particle: Particle):
        """
        Observe a particle entering the system.
        Build associations with recent context.
        """
        # Index the content
        content_key = str(particle.raw_value).lower()
        self.content_index[content_key].append(particle.id)
        self.domain_content[particle.domain_id].append(particle.raw_value)
        
        # Build associations with everything in context window
        for ctx_domain, ctx_pid, ctx_value in self.context_window:
            # Domain-to-domain association
            self._associate(particle.domain_id, ctx_domain, strength=0.3)
            
            # Content-to-content association (stronger for same-sentence)
            self._associate(
                f"content:{content_key}",
                f"content:{str(ctx_value).lower()}",
                strength=0.5
            )
            
            # Domain-to-content cross-association
            self._associate(
                particle.domain_id,
                f"content:{str(ctx_value).lower()}",
                strength=0.2
            )
        
        # Add to context window
        self.context_window.append((particle.domain_id, particle.id, particle.raw_value))
        if len(self.context_window) > self.window_size:
            self.context_window.pop(0)
    
    def _associate(self, source: str, target: str, strength: float = 0.3):
        """Create or reinforce an association."""
        if source == target:
            return
        
        # Forward association
        if target in self.associations[source]:
            self.associations[source][target].reinforce(strength * 0.5)
        else:
            self.associations[source][target] = Association(
                source=source,
                target=target,
                strength=strength
            )
            self.total_associations += 1
        
        # Reverse index
        if source not in self.reverse_index[target]:
            self.reverse_index[target].append(source)
        
        # Bidirectional (weaker)
        if source in self.associations[target]:
            self.associations[target][source].reinforce(strength * 0.3)
        else:
            self.associations[target][source] = Association(
                source=target,
                target=source,
                strength=strength * 0.5
            )
            self.total_associations += 1
        
        if target not in self.reverse_index[source]:
            self.reverse_index[source].append(target)
    
    def retrieve(self, query: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        Retrieve items associated with query.
        
        Args:
            query: Search query (content or domain)
            top_k: Maximum results
        
        Returns:
            List of (item, strength) sorted by strength
        """
        self.total_retrievals += 1
        
        results: Dict[str, float] = defaultdict(float)
        
        # Normalize query
        query_lower = query.lower()
        
        # Direct content match
        content_key = f"content:{query_lower}"
        if content_key in self.associations:
            for target, assoc in self.associations[content_key].items():
                results[target] += assoc.strength * 2.0  # Boost direct matches
        
        # Partial content match
        for indexed_content in self.content_index.keys():
            if query_lower in indexed_content or indexed_content in query_lower:
                content_key = f"content:{indexed_content}"
                if content_key in self.associations:
                    for target, assoc in self.associations[content_key].items():
                        results[target] += assoc.strength
        
        # Domain match
        for domain_id in self.domain_content.keys():
            if query_lower in domain_id.lower():
                if domain_id in self.associations:
                    for target, assoc in self.associations[domain_id].items():
                        results[target] += assoc.strength * 1.5
        
        # Check domain contents for matches
        for domain_id, contents in self.domain_content.items():
            for content in contents:
                if query_lower in str(content).lower():
                    # Found query in this domain - get associated domains
                    if domain_id in self.associations:
                        for target, assoc in self.associations[domain_id].items():
                            results[target] += assoc.strength
        
        # Sort by strength
        sorted_results = sorted(results.items(), key=lambda x: -x[1])
        
        return sorted_results[:top_k]
    
    def recall(self, query: str, identity_domains: Dict) -> List[Any]:
        """
        Recall actual content associated with query.
        
        Args:
            query: Search query
            identity_domains: Domains from identity
        
        Returns:
            List of recalled raw values
        """
        # Get associated items
        associations = self.retrieve(query, top_k=20)
        
        recalled = []
        seen = set()
        
        for item, strength in associations:
            if strength < 0.1:
                continue
            
            # Extract content from association
            if item.startswith("content:"):
                content = item[8:]  # Remove "content:" prefix
                if content not in seen and content != query.lower():
                    recalled.append((content, strength))
                    seen.add(content)
            
            # Get domain contents
            elif item in identity_domains:
                domain = identity_domains[item]
                for particle in list(domain.particles.values())[:5]:
                    val = str(particle.raw_value)
                    if val.lower() not in seen and val.lower() != query.lower():
                        recalled.append((val, strength * 0.8))
                        seen.add(val.lower())
        
        # Sort by strength and return values
        recalled.sort(key=lambda x: -x[1])
        return [val for val, _ in recalled[:10]]
    
    def find_completion(self, prefix: str, identity_domains: Dict) -> Optional[str]:
        """
        Find the most likely completion for a prefix based on associations.
        
        Used for answering questions like "Your name is ___"
        """
        # Get strongest associations
        associations = self.retrieve(prefix, top_k=30)
        
        # Look for content associations
        for item, strength in associations:
            if item.startswith("content:"):
                content = item[8:]
                # Skip if it's part of the query
                if content in prefix.lower():
                    continue
                # Skip common words
                if content in ['is', 'the', 'a', 'an', 'your', 'what', 'my']:
                    continue
                if len(content) > 1:  # Skip single chars
                    return content
        
        return None
    
    def get_stats(self) -> Dict:
        return {
            "total_associations": self.total_associations,
            "total_retrievals": self.total_retrievals,
            "indexed_content": len(self.content_index),
            "domains_tracked": len(self.domain_content),
            "context_window_size": len(self.context_window)
        }


class Retriever:
    """
    High-level retrieval interface.
    Combines associative memory with identity for meaningful recall.
    """
    
    def __init__(self, memory: AssociativeMemory, identity):
        self.memory = memory
        self.identity = identity
    
    def query(self, question: str) -> str:
        """
        Answer a question using associative recall.
        
        Args:
            question: Natural language question
        
        Returns:
            Best answer found, or indication of no answer
        """
        # Extract key terms from question
        question_lower = question.lower()
        
        # Remove question words
        skip_words = {'what', 'is', 'your', 'my', 'the', 'a', 'an', 'who', 'where', 'when', 'how', 'why', '?'}
        terms = [w for w in question_lower.split() if w not in skip_words and len(w) > 1]
        
        if not terms:
            return "(no key terms found)"
        
        # Query each term and combine results
        all_results: Dict[str, float] = defaultdict(float)
        
        for term in terms:
            results = self.memory.retrieve(term, top_k=20)
            for item, strength in results:
                all_results[item] += strength
        
        if not all_results:
            return "(no associations found)"
        
        # Find best content match
        best_content = None
        best_strength = 0
        
        for item, strength in sorted(all_results.items(), key=lambda x: -x[1]):
            if item.startswith("content:"):
                content = item[8:]
                # Skip if in question
                if content in question_lower:
                    continue
                # Skip common words
                if content in skip_words:
                    continue
                if len(content) > 1 and strength > best_strength:
                    best_content = content
                    best_strength = strength
        
        if best_content:
            return best_content
        
        # Try domain-based recall
        recalled = self.memory.recall(" ".join(terms), self.identity.domains)
        if recalled:
            # Filter out question words
            filtered = [r for r in recalled if r.lower() not in question_lower]
            if filtered:
                return filtered[0]
        
        return "(no answer found)"
    
    def associate_statement(self, statement: str):
        """
        Process a statement to build strong associations.
        
        For statements like "Your name is PBAI", creates strong
        association between "name" and "PBAI".
        """
        words = statement.lower().split()
        
        # Look for "is" patterns: "X is Y"
        if 'is' in words:
            is_idx = words.index('is')
            before = words[:is_idx]
            after = words[is_idx + 1:]
            
            # Create strong associations between before and after
            for b in before:
                if len(b) > 1:
                    for a in after:
                        if len(a) > 1:
                            self.memory._associate(
                                f"content:{b}",
                                f"content:{a}",
                                strength=0.8  # Strong association for explicit statements
                            )
        
        # Look for possession patterns
        if 'your' in words or 'my' in words:
            # Find the noun after your/my
            for i, w in enumerate(words):
                if w in ['your', 'my'] and i + 1 < len(words):
                    noun = words[i + 1]
                    # Associate noun with everything after "is"
                    if 'is' in words:
                        is_idx = words.index('is')
                        for a in words[is_idx + 1:]:
                            if len(a) > 1:
                                self.memory._associate(
                                    f"content:{noun}",
                                    f"content:{a}",
                                    strength=0.9  # Very strong for "your X is Y"
                                )
