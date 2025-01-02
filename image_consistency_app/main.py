from fastapi import FastAPI, HTTPException
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
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
import json

from models import ChatRequest, ChatResponse, ImageMemory
from memory_manager import ImageMemoryManager
from utils import generate_image_with_consistency

# Load environment variables
load_dotenv()

HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
if not HUGGINGFACE_API_TOKEN:
    raise EnvironmentError("HUGGINGFACE_API_TOKEN environment variable is not set")

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
# try:
#     # Initialize LLM
#     text_generator = pipeline(
#         "text-generation",
#         model=LLM_MODEL_NAME,
#         device_map="auto"  # Automatically choose best device
#     )
    
#     # Initialize Image Generator
#     image_generator = StableDiffusionPipeline.from_pretrained(
#         IMAGE_MODEL_NAME,
#         torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
#         safety_checker=None  # Disable safety checker for speed
#     )
    
#     if torch.cuda.is_available():
#         image_generator = image_generator.to("cuda")
        
# except Exception as e:
#     logger.error(f"Error initializing models: {str(e)}")
#     raise

# Initialize memory manager
memory_manager = ImageMemoryManager()

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        user_input = request.user_input
        context = request.context
        
        # Generate LLM response with function calling
        prompt = f"""<s>[INST] <<SYS>>\n{CHARACTER_PERSONA}\n\nYou have access to the `generate_image` function. Use it when the user asks for a visual representation. 
        The function takes a single argument `prompt` which is a string describing the image to generate. 
        If the user asks for an image, 
        respond with the following 
        JSON format: 
        {{
            "function_call": 
            {{
            "name": "generate_image", "arguments": 
            {{
                "prompt": "image description"
            }}    
            }}
        }}
        . If the user does not ask for an image, respond as a normal chatbot.\n<</SYS>>\n\n{user_input} [/INST]"""
        
        # Use Hugging Face API for LLM
        headers = {"Authorization": f"Bearer {HUGGINGFACE_API_TOKEN}"}
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_length": 500,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "num_return_sequences": 1,
                "pad_token_id": 2,
                "truncation": True
            }
        }
        
        response = requests.post(f"https://api-inference.huggingface.co/models/{LLM_MODEL_NAME}", headers=headers, json=payload)
        response.raise_for_status()
        llm_response = response.json()[0]['generated_text']
        
        # Clean up response
        llm_response = llm_response.split("[/INST]")[-1].strip()
        
        # Check if function call is needed
        debug_info = {"image_generation_triggered": False}
        image_url = None
        
        try:
            function_call = json.loads(llm_response).get("function_call")
            if function_call and function_call["name"] == "generate_image":
                debug_info["image_generation_triggered"] = True
                image_prompt = function_call["arguments"]["prompt"]
                
                # Use Together AI API for image generation
                TOGETHER_API_KEY = os.getenv("TOGETHER_API_KEY")
                logger.info(f"Using Together API key: {TOGETHER_API_KEY}")
                
                if not TOGETHER_API_KEY:
                    raise EnvironmentError("TOGETHER_API_KEY environment variable is not set")
                
                headers = {
                    "Authorization": f"Bearer {TOGETHER_API_KEY}",
                    "Content-Type": "application/json"
                }
                payload = {
                    "model": "black-forest-labs/FLUX.1-schnell-Free",
                    "prompt": image_prompt,
                    "steps": 20,
                    "guidance_scale": 7.5
                }
                
                response = requests.post("https://api.together.xyz/inference", headers=headers, json=payload)
                response.raise_for_status()
                image_data = response.json()['output']['images'][0]
                image_url = f"data:image/jpeg;base64,{image_data}"
        except:
            pass
            
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
