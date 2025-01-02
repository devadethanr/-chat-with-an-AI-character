import re
from PIL import Image
import logging
from typing import Optional
from io import BytesIO
import base64
import torch
from datetime import datetime
from models import ImageMemory
from memory_manager import ImageMemoryManager

logger = logging.getLogger(__name__)

def should_generate_image(text: str) -> bool:
    """
    Enhanced detection of image generation triggers using context and keywords.
    """
    # Keywords that might indicate image generation is needed
    visual_triggers = {
        'show', 'draw', 'sketch', 'paint', 'picture', 'image', 'illustration',
        'visualize', 'demonstrate', 'display', 'present', 'look', 'see'
    }
    
    # Context patterns that might indicate visual content
    visual_contexts = [
        r'what (does|did|would) .* look like',
        r'can (you |I |we )?(see|view|look at)',
        r'(show|draw|paint|sketch) .* for me',
        r'make .* (drawing|sketch|painting)',
        r'visualize .*',
    ]
    
    # Check for direct keywords
    words = set(text.lower().split())
    if any(trigger in words for trigger in visual_triggers):
        return True
        
    # Check for context patterns
    if any(re.search(pattern, text.lower()) for pattern in visual_contexts):
        return True
        
    return False

def extract_image_prompt(text: str) -> str:
    """
    Enhanced extraction of image prompts from text.
    """
    # Common patterns for image requests
    patterns = [
        r'show me (.*?)(?:\.|$)',
        r'draw (.*?)(?:\.|$)',
        r'paint (.*?)(?:\.|$)',
        r'sketch (.*?)(?:\.|$)',
        r'visualize (.*?)(?:\.|$)',
        r'picture of (.*?)(?:\.|$)',
        r'image of (.*?)(?:\.|$)',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
            
    # If no pattern matches, use the whole text
    return text.strip()

def generate_image_with_consistency(
    image_generator,
    memory_manager: ImageMemoryManager,
    prompt: str,
    context: str,
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5
) -> Image.Image:
    """
    Generate image while maintaining consistency with previous generations.
    """
    try:
        # Check for relevant previous images
        relevant_memory = memory_manager.find_relevant_memory(prompt, context)
        
        if relevant_memory:
            # Enhance prompt with consistent elements
            enhanced_prompt = f"Consistent with previous scene: {relevant_memory.description}. New scene: {prompt}"
            logger.info(f"Enhanced prompt with consistency: {enhanced_prompt}")
        else:
            enhanced_prompt = prompt
            
        # Generate image with enhanced parameters
        image = image_generator(
            enhanced_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale
        ).images[0]
        
        return image
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise