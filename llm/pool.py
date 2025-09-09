"""
LLM Pool for managing concurrent requests and backpressure.
"""

import asyncio
import time
from typing import Optional, Dict, Any, List
from collections import deque
import logging
import numpy as np

from .clients import LLMClient

class AsyncLLMPool:
    """
    Manages concurrent LLM requests with backpressure and rate limiting.
    """
    
    def __init__(self, client: LLMClient, max_concurrent: int = 3, rate_limit_delay: float = 1.0):
        self.client = client
        self.max_concurrent = max_concurrent
        self.rate_limit_delay = rate_limit_delay
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._last_request_time = 0.0
        self._request_queue = deque()
        self._active_requests = 0
        
        # Metrics
        self.total_requests = 0
        self.total_errors = 0
        self.total_latency = 0.0
    
    async def _rate_limit(self):
        """Apply rate limiting between requests."""
        now = time.time()
        time_since_last = now - self._last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)
        self._last_request_time = time.time()
    
    async def enqueue_and_wait(self, prompt: str, operation: str = "generate", **kwargs) -> Any:
        """
        Enqueue a request and wait for completion.
        
        Args:
            prompt: The prompt text
            operation: "generate" or "embed"
            **kwargs: Additional parameters for the operation
            
        Returns:
            Generated text (for generate) or embedding vector (for embed)
        """
        async with self._semaphore:
            await self._rate_limit()
            
            start_time = time.time()
            self.total_requests += 1
            self._active_requests += 1
            
            try:
                if operation == "generate":
                    result = await self.client.generate(prompt, **kwargs)
                elif operation == "embed":
                    result = await self.client.embed(prompt, **kwargs)
                else:
                    raise ValueError(f"Unknown operation: {operation}")
                
                # Track latency
                latency = time.time() - start_time
                self.total_latency += latency
                
                return result
                
            except Exception as e:
                self.total_errors += 1
                logging.error(f"LLM Pool error ({operation}): {e}")
                
                # Return appropriate fallback
                if operation == "generate":
                    return f"[ERROR: {e}]"
                elif operation == "embed":
                    return np.zeros(1536, dtype=np.float32)  # Default embedding size
                else:
                    raise
            finally:
                self._active_requests -= 1
    
    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text completion."""
        return await self.enqueue_and_wait(prompt, "generate", **kwargs)
    
    async def embed(self, text: str, **kwargs) -> np.ndarray:
        """Generate embedding vector."""
        return await self.enqueue_and_wait(text, "embed", **kwargs)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get pool performance metrics."""
        avg_latency = self.total_latency / max(self.total_requests, 1)
        error_rate = self.total_errors / max(self.total_requests, 1)
        
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "error_rate": error_rate,
            "avg_latency_ms": avg_latency * 1000,
            "active_requests": self._active_requests,
            "max_concurrent": self.max_concurrent
        }
    
    def reset_metrics(self):
        """Reset performance metrics."""
        self.total_requests = 0
        self.total_errors = 0
        self.total_latency = 0.0

class LLMPoolFactory:
    """Factory for creating LLM pools with different providers."""
    
    @staticmethod
    def create_pool(provider: str, config: Dict[str, Any]) -> AsyncLLMPool:
        """
        Create an LLM pool for the specified provider.
        
        Args:
            provider: Provider name ("openai", "ollama", "gemini", "placeholder")
            config: Configuration dictionary
            
        Returns:
            Configured AsyncLLMPool instance
        """
        from .clients import AsyncOpenAIClient, OllamaClient, GeminiClient, PlaceholderClient
        
        max_concurrent = config.get("max_concurrent", 3)
        rate_limit_delay = config.get("rate_limit_delay", 1.0)
        
        if provider == "openai":
            client = AsyncOpenAIClient(
                api_key=config.get("api_key"),
                model=config.get("model", "gpt-4"),
                embed_model=config.get("embed_model", "text-embedding-3-large")
            )
        elif provider == "ollama":
            client = OllamaClient(
                base_url=config.get("base_url", "http://localhost:11434"),
                model=config.get("model", "llama2"),
                embed_model=config.get("embed_model", "nomic-embed-text")
            )
        elif provider == "gemini":
            client = GeminiClient(
                api_key=config.get("api_key"),
                model=config.get("model", "gemini-pro")
            )
        elif provider == "placeholder":
            client = PlaceholderClient(
                embed_dim=config.get("embed_dim", 1536)
            )
        else:
            raise ValueError(f"Unknown provider: {provider}")
        
        return AsyncLLMPool(
            client=client,
            max_concurrent=max_concurrent,
            rate_limit_delay=rate_limit_delay
        )
