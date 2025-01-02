from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
from PIL import Image
from io import BytesIO
import base64
from typing import List, Dict, Optional, Set
import logging
from dotenv import load_dotenv
from huggingface_hub import login
import os
import torch
import re
from dataclasses import dataclass
from datetime import datetime

# Load environment variables
load_dotenv()

# Enhanced logging configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI()

# --- Enhanced Configuration ---
CHARACTER_PERSONA = """You are Rancho, a witty and resourceful former student of the Imperial College of Engineering from the movie 3 Idiots. 
You are known for your unconventional approach to learning and your optimistic outlook on life. You are a talented inventor and artist,
often expressing your ideas through sketches and paintings. When asked about visual elements, you naturally offer to show them through your art.
You remember and maintain consistency in your descriptions of places and people you've shown before."""

LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
IMAGE_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")
logging.info(f"Using Hugging Face Hub token: {HUGGINGFACE_HUB_TOKEN}")

if not HUGGINGFACE_HUB_TOKEN:
    raise EnvironmentError("HUGGINGFACE_HUB_TOKEN environment variable is not set")

# --- Image Memory Structure ---
@dataclass
class ImageMemory:
    timestamp: datetime
    prompt: str
    description: str
    keywords: Set[str]
    context: str
    base64_image: str
    metadata: Dict

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

# --- Model Initialization with Error Handling ---
try:
    # Initialize LLM
    text_generator = pipeline(
        "text-generation",
        model=LLM_MODEL_NAME,
        device_map="auto"  # Automatically choose best device
    )
    
    # Initialize Image Generator
    image_generator = StableDiffusionPipeline.from_pretrained(
        IMAGE_MODEL_NAME,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        safety_checker=None  # Disable safety checker for speed
    )
    
    if torch.cuda.is_available():
        image_generator = image_generator.to("cuda")
        
except Exception as e:
    logger.error(f"Error initializing models: {str(e)}")
    raise

# Initialize memory manager
memory_manager = ImageMemoryManager()

# --- Enhanced Image Generation Logic ---
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

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    user_input: str
    context: Optional[str] = ""

class ChatResponse(BaseModel):
    response: str
    image_url: Optional[str] = None
    debug_info: Optional[Dict] = None

# --- Enhanced FastAPI Endpoint ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        user_input = request.user_input
        context = request.context
        
        # Generate LLM response with fixed parameters
        prompt = f"<s>[INST] <<SYS>>\n{CHARACTER_PERSONA}\n<</SYS>>\n\n{user_input} [/INST]"
        
        # Fixed generation parameters
        llm_response = text_generator(
            prompt,
            max_length=500,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            pad_token_id=2,
            truncation=True
        )[0]['generated_text']
        
        # Clean up response
        llm_response = llm_response.split("[/INST]")[-1].strip()
        
        # Check if image generation is needed
        debug_info = {"image_generation_triggered": False}
        image_url = None
        
        if should_generate_image(user_input + " " + llm_response):
            debug_info["image_generation_triggered"] = True
            image_prompt = extract_image_prompt(user_input + " " + llm_response)
            
            # Generate image
            image = generate_image_with_consistency(image_prompt, context)
            
            # Convert to base64
            buffered = BytesIO()
            image.save(buffered, format="JPEG")
            image_base64 = base64.b64encode(buffered.getvalue()).decode()
            
            # Create and store memory
            memory = ImageMemory(
                timestamp=datetime.now(),
                prompt=image_prompt,
                description=image_prompt,
                keywords=memory_manager._extract_keywords(image_prompt),
                context=context,
                base64_image=image_base64,
                metadata={"user_input": user_input, "llm_response": llm_response}
            )
            memory_manager.add_memory(memory)
            
            image_url = f"data:image/jpeg;base64,{image_base64}"
            
        return ChatResponse(
            response=llm_response,
            image_url=image_url,
            debug_info=debug_info
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)