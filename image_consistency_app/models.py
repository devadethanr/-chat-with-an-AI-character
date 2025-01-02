from pydantic import BaseModel
from typing import Dict, Optional
from dataclasses import dataclass
from datetime import datetime
from typing import Set

class ChatRequest(BaseModel):
    user_input: str
    context: Optional[str] = ""

class ChatResponse(BaseModel):
    response: str
    image_url: Optional[str] = None
    debug_info: Optional[Dict] = None

@dataclass
class ImageMemory:
    timestamp: datetime
    prompt: str
    description: str
    keywords: Set[str]
    context: str
    base64_image: str
    metadata: Dict
