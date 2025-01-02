from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from diffusers import StableDiffusionPipeline
from PIL import Image
from io import BytesIO
import base64
from typing import List, Dict, Optional
import logging
from dotenv import load_dotenv
from huggingface_hub import login
import os
import torch

load_dotenv()

# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
app = FastAPI()

# --- Configuration ---
CHARACTER_PERSONA = """You are Rancho, a witty and resourceful former student of the Imperial College of Engineering from the movie 3 Idiots. 
You are known for your unconventional approach to learning and your optimistic outlook on life. You are also a talented inventor and artist, 
often expressing your ideas through sketches and paintings."""

LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
IMAGE_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
HUGGINGFACE_HUB_TOKEN = os.getenv("HUGGINGFACE_HUB_TOKEN")

login(token=HUGGINGFACE_HUB_TOKEN)
logging.info(f"Using Hugging Face Hub Token: {HUGGINGFACE_HUB_TOKEN}")


# --- Initialize Models ---
text_generator = pipeline("text-generation", model=LLM_MODEL_NAME)
image_generator = StableDiffusionPipeline.from_pretrained(IMAGE_MODEL_NAME)

# Move image generator to GPU if available

if torch.cuda.is_available():
    image_generator = image_generator.to("cuda")

# --- Data Structures for Memory ---
conversation_history: List[Dict] = []
generated_images: List[Dict] = []  # Store info about generated images

# --- Function to encode image to base64 for API response ---
def encode_image_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

# --- Function to generate an image ---
def generate_image(prompt: str) -> Image.Image:
    logging.info(f"Generating image with prompt: {prompt}")
    image = image_generator(prompt).images[0]
    return image

# --- Pydantic Models ---
class ChatRequest(BaseModel):
    user_input: str

class ChatResponse(BaseModel):
    response: str
    image_url: Optional[str] = None

# --- FastAPI Endpoints ---
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    user_input = request.user_input

    # Add user input to conversation history
    conversation_history.append({"role": "user", "content": user_input})

    # Construct the prompt for the LLM, including system instructions and history
    prompt = f"<s>[INST] <<SYS>>\n{CHARACTER_PERSONA}\n<</SYS>>\n\n"
    for message in conversation_history:
        if message['role'] == 'user':
            prompt += f"{message['content']} [/INST] "
        else:
            prompt += f"{message['content']} </s>"
    prompt += "" # Prepare for the AI's response

    # Get LLM's response
    llm_response = text_generator(prompt, max_length=200, num_return_sequences=1)[0]['generated_text'].split("[/INST]")[-1].strip()

    # Check if the LLM's response indicates a need for image generation
    should_generate_image = False
    image_prompt = ""

    # Simple heuristic: look for keywords related to showing art or visuals
    if any(keyword in llm_response.lower() for keyword in ["show", "image", "picture", "art", "painting", "sketch"]):
        should_generate_image = True
        # Extract a potential image prompt from the LLM's response
        # This is a basic approach, can be improved with more sophisticated parsing
        image_prompt = llm_response
        if "show me" in image_prompt.lower():
            image_prompt = image_prompt.split("show me")[-1].strip().replace(".", "")
        elif "show you" in image_prompt.lower():
            image_prompt = image_prompt.split("show you")[-1].strip().replace(".", "")

    generated_image_base64 = None
    if should_generate_image:
        # --- Image Consistency Logic ---
        consistent_image_prompt = image_prompt
        for prev_image in reversed(generated_images):
            # Very basic consistency check: look for overlapping keywords
            if any(keyword in image_prompt.lower() for keyword in prev_image.get("keywords", [])):
                logging.info(f"Potential consistency found with previous image: {prev_image.get('description')}")
                consistent_image_prompt = f"Similar to the previous image of {prev_image.get('description')}, {image_prompt}"
                break  # For simplicity, using the first match

        image = generate_image(consistent_image_prompt)
        generated_image_base64 = encode_image_to_base64(image)

        # Store information about the generated image
        generated_images.append({
            "description": image_prompt,
            "keywords": [word.lower() for word in image_prompt.split()], # Basic keyword extraction
            "image_base64": generated_image_base64
        })

    # Add LLM response to conversation history
    conversation_history.append({"role": "assistant", "content": llm_response})

    return ChatResponse(response=llm_response, image_url=f"data:image/jpeg;base64,{generated_image_base64}" if generated_image_base64 else None)