from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class DocumentChunk:
    content: str
    metadata: Dict[str, Any]
