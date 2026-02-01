import os

from openai import OpenAI, OpenAIError


def embed_texts(texts: list[str]) -> list[list[float]]:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise OpenAIError(
            "OPENAI_API_KEY is not set. Export it or set it in your .env file."
        )
    client = OpenAI(api_key=api_key)
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]
