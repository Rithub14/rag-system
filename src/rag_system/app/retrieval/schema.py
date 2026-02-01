from typing import Dict, Any
from pydantic import BaseModel


class DocumentChunk(BaseModel):
    content: str
    metadata: Dict[str, Any]
