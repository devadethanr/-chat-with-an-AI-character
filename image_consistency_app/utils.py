from PIL import Image
import logging
import os
from datetime import datetime
from typing import Optional
from models import ImageMemory
from memory_manager import ImageMemoryManager

logger = logging.getLogger(__name__)

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
        
        # Save the generated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image_dir = "generated_images"
        os.makedirs(image_dir, exist_ok=True)
        image_path = os.path.join(image_dir, f"generated_image_{timestamp}.png")
        image.save(image_path)
        logger.info(f"Image saved to {image_path}")
        
        return image
        
    except Exception as e:
        logger.error(f"Error generating image: {str(e)}")
        raise
