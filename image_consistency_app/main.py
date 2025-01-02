from fastapi import FastAPI, HTTPException
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
from PIL import Image
from io import BytesIO
import base64
import logging
from dotenv import load_dotenv
from huggingface_hub import login
import os
import torch
from datetime import datetime

from models import ChatRequest, ChatResponse, ImageMemory
from memory_manager import ImageMemoryManager
from utils import should_generate_image, extract_image_prompt, generate_image_with_consistency

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
logger.info(f"Using Hugging Face Hub token: {HUGGINGFACE_HUB_TOKEN}")

if not HUGGINGFACE_HUB_TOKEN:
    raise EnvironmentError("HUGGINGFACE_HUB_TOKEN environment variable is not set")

# Initialize models
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
            image = generate_image_with_consistency(
                image_generator,
                memory_manager,
                image_prompt,
                context
            )
            
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