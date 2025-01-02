from typing import List, Optional, Set, Dict
import re
from datetime import datetime
from models import ImageMemory

class ImageMemoryManager:
    def __init__(self):
        self.memories: List[ImageMemory] = []
        
    def add_memory(self, memory: ImageMemory):
        self.memories.append(memory)
        
    def find_relevant_memory(self, prompt: str, context: str) -> Optional[ImageMemory]:
        if not self.memories:
            return None
            
        # Convert prompt to keywords
        prompt_keywords = set(self._extract_keywords(prompt))
        
        # Score each memory based on keyword overlap and context similarity
        best_match = None
        highest_score = 0
        
        for memory in self.memories:
            # Calculate keyword overlap score
            keyword_overlap = len(prompt_keywords.intersection(memory.keywords)) / len(prompt_keywords) if prompt_keywords else 0
            
            # Calculate context similarity (simple string matching for now)
            context_similarity = self._calculate_context_similarity(context, memory.context)
            
            # Combined score
            score = (keyword_overlap * 0.7) + (context_similarity * 0.3)
            
            if score > highest_score and score > 0.3:  # Threshold for relevance
                highest_score = score
                best_match = memory
                
        return best_match
    
    @staticmethod
    def _extract_keywords(text: str) -> Set[str]:
        # Enhanced keyword extraction
        words = re.findall(r'\b\w+\b', text.lower())
        # Remove common words and keep meaningful ones
        stopwords = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        return {word for word in words if word not in stopwords}
    
    @staticmethod
    def _calculate_context_similarity(context1: str, context2: str) -> float:
        # Simple context similarity based on common words
        words1 = set(context1.lower().split())
        words2 = set(context2.lower().split())
        return len(words1.intersection(words2)) / max(len(words1), len(words2))
