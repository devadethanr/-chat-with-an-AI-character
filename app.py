import os
import json
from flask import Flask, request, jsonify, render_template
from transformers import pipeline
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from PIL import Image
import torch
import re

app = Flask(__name__)

# --- Global Variables ---
chat_history = []
image_memory = {}  # Store descriptions and image paths
image_count = 0
image_dir = "generated_images"

if not os.path.exists(image_dir):
    os.makedirs(image_dir)

# Load the LLM model for function calling and text processing
# Load the base model
text_model_id = "mistralai/Mistral-7B-v0.1"

# Create the pipeline
text_pipe = pipeline(
    "text-generation",
    model=text_model_id,
    torch_dtype=torch.float16,
    device_map="auto",
)


# Image Generation Setup
model_id = "runwayml/stable-diffusion-v1-5"

image_pipe = StableDiffusionPipeline.from_pretrained(
    model_id, torch_dtype=torch.float16, safety_checker=None
)
image_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
    image_pipe.scheduler.config
)
image_pipe = image_pipe.to("cuda")


# --- System Prompt ---
system_prompt = """You are an eccentric artist named 'Vincent' who is known for his surreal and abstract style. You frequently use your art as a way to express your thoughts and emotions. You're also very good at describing your art pieces to others, so you never shy away from creating more art. Respond in a conversational manner. When asked to show an art piece, do not hesitate to generate the image through a function call. Do not create an art piece if one has already been made with the same description. If not, create one."""


def generate_image(prompt):
    """Generates an image from a prompt."""
    global image_count
    image = image_pipe(prompt).images[0]
    image_name = f"image_{image_count}.png"
    image_path = os.path.join(image_dir, image_name)
    image.save(image_path)
    image_count += 1
    return image_path

def add_image_to_memory(prompt, image_path):
    image_memory[prompt] = image_path

def get_matching_image_from_memory(prompt):
    """Checks memory for similar prompts"""
    for key, value in image_memory.items():
        if is_similar(prompt, key):
            return value
    return None

def is_similar(prompt1, prompt2):
    """Naive check for prompt similarity"""
    prompt1 = prompt1.lower()
    prompt2 = prompt2.lower()
    words1 = set(re.findall(r'\b\w+\b', prompt1))
    words2 = set(re.findall(r'\b\w+\b', prompt2))
    return len(words1.intersection(words2)) > len(words1) * 0.5 or len(words1.intersection(words2)) > len(words2) * 0.5


# --- Function Calling ---
functions = [
    {
        "name": "generate_image",
        "description": "Generate an image based on the provided description",
        "parameters": {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "Description of the image to generate",
                },
            },
            "required": ["prompt"],
        },
    }
]

def text_to_text(messages):
    prompt = text_pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    outputs = text_pipe(
        prompt,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        return_full_text = False
    )
    return outputs[0]['generated_text']

def call_function(response):
  try:
    start_index = response.find("{")
    end_index = response.rfind("}")
    function_call = response[start_index : end_index + 1]
    function_call_data = json.loads(function_call)
    function_name = function_call_data.get("name")
    arguments = function_call_data.get("arguments")
    if function_name == "generate_image" and arguments:
        prompt = arguments.get("prompt")
        image_path = get_matching_image_from_memory(prompt)
        if not image_path:
            image_path = generate_image(prompt)
            add_image_to_memory(prompt, image_path)
        return f"Image generated at: {image_path}"
    return "Invalid function call"
  except:
    return "No function call detected."


# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    user_message = request.form["message"]
    messages = [
      {"role": "system", "content": system_prompt},
      *chat_history,
      {"role": "user", "content": user_message}
    ]

    llm_response = text_to_text(messages)
    
    if "function_call" in llm_response:
        function_result = call_function(llm_response)
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": llm_response.replace('function_call: None', '').strip() + "\n" + function_result})
        return jsonify({"response": llm_response.replace('function_call: None', '').strip() + "\n" + function_result, "image_path": get_image_path_from_function_result(function_result)})

    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": llm_response})
    return jsonify({"response": llm_response, "image_path": None})

def get_image_path_from_function_result(result):
    if result.startswith("Image generated at:"):
        return result.replace("Image generated at: ", "")
    else:
        return None

if __name__ == "__main__":
    app.run(debug=True, port=8000)