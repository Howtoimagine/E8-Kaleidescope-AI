"""
LLM client implementations for E8Mind system.
Provides unified interface for different LLM providers.
"""

import os
import asyncio
import logging
import numpy as np
from typing import Protocol, Optional, List, Dict, Any
from abc import ABC, abstractmethod

class LLMClient(Protocol):
    """Protocol for LLM client implementations."""
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion from prompt."""
        ...
    
    async def embed(self, text: str, **kwargs) -> np.ndarray:
        """Generate embedding vector from text."""
        ...

class AsyncOpenAIClient:
    """OpenAI API client with async support."""
    
    def __init__(self, api_key: str = None, model: str = "gpt-4", embed_model: str = "text-embedding-3-large"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.embed_model = embed_model
        self._client = None
        
        if not self.api_key:
            raise ValueError("OpenAI API key is required")
    
    async def _get_client(self):
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                self._client = AsyncOpenAI(api_key=self.api_key)
            except ImportError:
                raise ImportError("openai package is required for OpenAI client")
        return self._client
    
    async def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7, **kwargs) -> str:
        """Generate text completion using OpenAI API."""
        try:
            client = await self._get_client()
            response = await client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
                **kwargs
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            logging.error(f"OpenAI generate error: {e}")
            return f"[ERROR: {e}]"
    
    async def embed(self, text: str, **kwargs) -> np.ndarray:
        """Generate embedding using OpenAI API."""
        try:
            client = await self._get_client()
            response = await client.embeddings.create(
                model=self.embed_model,
                input=text,
                **kwargs
            )
            return np.array(response.data[0].embedding, dtype=np.float32)
        except Exception as e:
            logging.error(f"OpenAI embed error: {e}")
            # Return zero vector as fallback
            return np.zeros(1536, dtype=np.float32)

class OllamaClient:
    """Ollama local LLM client."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama2", embed_model: str = "nomic-embed-text"):
        self.base_url = base_url
        self.model = model
        self.embed_model = embed_model
    
    async def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7, **kwargs) -> str:
        """Generate text completion using Ollama."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "options": {
                        "num_predict": max_tokens,
                        "temperature": temperature,
                        **kwargs
                    },
                    "stream": False
                }
                async with session.post(f"{self.base_url}/api/generate", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return data.get("response", "")
                    else:
                        return f"[ERROR: HTTP {response.status}]"
        except Exception as e:
            logging.error(f"Ollama generate error: {e}")
            return f"[ERROR: {e}]"
    
    async def embed(self, text: str, **kwargs) -> np.ndarray:
        """Generate embedding using Ollama."""
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                payload = {
                    "model": self.embed_model,
                    "prompt": text,
                    **kwargs
                }
                async with session.post(f"{self.base_url}/api/embeddings", json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        return np.array(data.get("embedding", []), dtype=np.float32)
                    else:
                        # Return zero vector as fallback
                        return np.zeros(384, dtype=np.float32)  # Common embedding dim for local models
        except Exception as e:
            logging.error(f"Ollama embed error: {e}")
            return np.zeros(384, dtype=np.float32)

class GeminiClient:
    """Google Gemini API client."""
    
    def __init__(self, api_key: str = None, model: str = "gemini-pro"):
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        self.model = model
        
        if not self.api_key:
            raise ValueError("Google API key is required")
    
    async def generate(self, prompt: str, max_tokens: int = 1000, temperature: float = 0.7, **kwargs) -> str:
        """Generate text completion using Gemini API."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            model = genai.GenerativeModel(self.model)
            
            # Run in thread pool since genai is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None, 
                lambda: model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        max_output_tokens=max_tokens,
                        temperature=temperature
                    )
                )
            )
            return response.text or ""
        except Exception as e:
            logging.error(f"Gemini generate error: {e}")
            return f"[ERROR: {e}]"
    
    async def embed(self, text: str, **kwargs) -> np.ndarray:
        """Generate embedding using Gemini API."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            # Run in thread pool since genai is synchronous
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: genai.embed_content(
                    model="models/embedding-001",
                    content=text,
                    task_type="retrieval_document"
                )
            )
            return np.array(response["embedding"], dtype=np.float32)
        except Exception as e:
            logging.error(f"Gemini embed error: {e}")
            return np.zeros(768, dtype=np.float32)  # Gemini embedding dimension

class PlaceholderClient:
    """Placeholder client for testing without real API calls."""
    
    def __init__(self, embed_dim: int = 1536):
        self.embed_dim = embed_dim
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Return a placeholder response."""
        return f"[PLACEHOLDER] Generated response for prompt: {prompt[:50]}..."
    
    async def embed(self, text: str, **kwargs) -> np.ndarray:
        """Return a random embedding vector."""
        # Generate deterministic embedding based on text hash
        import hashlib
        text_hash = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
        rng = np.random.default_rng(text_hash)
        vec = rng.standard_normal(self.embed_dim).astype(np.float32)
        return vec / np.linalg.norm(vec)
