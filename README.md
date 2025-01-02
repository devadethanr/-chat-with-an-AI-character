# AI Character Chat with Image Generation and Consistency

An AI-powered chat application featuring Rancho from "3 Idiots" that generates consistent images during conversations. The system maintains visual consistency across related images and automatically generates images when contextually appropriate.

## Features

- **Character-based Chat**: Interactive conversations with Rancho's persona from "3 Idiots"
- **Automatic Image Generation**: Contextually aware image creation without explicit commands
- **Image Consistency**: Maintains visual consistency across related images
- **Simple Web Interface**: Easy-to-use chat interface
- **Debug Information**: Visibility into the system's decision-making process

## Prerequisites

- Python 3.8+
- HuggingFace Account and API Token
- Sufficient disk space for ML models
- CUDA-capable GPU (recommended)

## Installation

1. Clone the repository:
```bash
git clone  ```https://github.com/devadethanr/chat-with-an-AI-character.git```
cd [repository-name]
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install fastapi uvicorn torch transformers diffusers pillow python-dotenv huggingface-hub
```

4. Create a `.env` file in the project root:
```env
HUGGINGFACE_HUB_TOKEN=your_token_here
```

## Project Structure

```
.
├── main.py              # FastAPI backend application
├── test_chat.html           # Simple web interface
├── requirements.txt    # Python dependencies
└── README.md          # This file
```

## Configuration

The application uses the following models:
- LLM: `mistralai/Mistral-7B-Instruct-v0.2`
- Image Generation: `runwayml/stable-diffusion-v1-5`

These can be configured in `main.py` by modifying the model constants:
```python
LLM_MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"
IMAGE_MODEL_NAME = "runwayml/stable-diffusion-v1-5"
```

## Running the Application

1. Start the backend server:
```bash
python main.py
```
The server will run on `http://localhost:8000`

2. Open another terminal:
- Simply run `test_chat.py`

## Testing

1. Basic conversation testing:
```python
python test_chat.py
```

## API Endpoints

### POST /chat
Creates a new chat message and generates a response.

Request body:
```json
{
  "user_input": "string",
  "context": "string"
}
```

Response:
```json
{
  "response": "string",
  "image_url": "string (optional)",
  "debug_info": "object (optional)"
}
```

## Technical Implementation Details

### Image Consistency System
- Uses an `ImageMemoryManager` to track generated images
- Maintains context and metadata for each image
- Uses keyword extraction and similarity matching for consistency

### Function Calling Integration
- Automatically detects need for image generation
- Uses context patterns and keywords
- Maintains conversation flow without explicit commands

### Memory Management
- Tracks conversation history
- Maintains image generation context
- Ensures consistent visual elements across related images

## Limitations

- Requires significant computational resources for image generation
- Image generation may take several seconds
- Limited to the knowledge cutoff date of the language model
- Image consistency is based on text descriptions and may vary

## Troubleshooting

Common issues and solutions:

1. CUDA/GPU issues:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

2. Memory issues:
- Reduce batch sizes
- Use CPU if GPU memory is insufficient
- Clear cache periodically

3. Model loading issues:
- Check internet connection
- Verify HuggingFace token
- Ensure sufficient disk space

###submitted to Bhabha AI, please feel free to pull the repo