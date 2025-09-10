"""
E8Mind LLM Client System - Extracted from M18.7

This module provides async LLM clients for OpenAI, Ollama, and Gemini APIs,
along with a connection pool for managing concurrent requests.
"""

import asyncio
import os
import json
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import traceback

# Default configuration
DEFAULT_OPENAI_CHAT_MODEL = "gpt-4o"
EMBED_DIM = 768
POOL_WORKER_TIMEOUT = 45.0

# Optional dependency imports
try:
    import openai
    from openai import AsyncOpenAI, BadRequestError
    OPENAI_AVAILABLE = True
except ImportError:
    openai = None  # type: ignore
    AsyncOpenAI = None  # type: ignore
    BadRequestError = None  # type: ignore
    OPENAI_AVAILABLE = False

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    ollama = None  # type: ignore
    OLLAMA_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    genai = None  # type: ignore
    GEMINI_AVAILABLE = False


class AsyncOpenAIClient:
    """Async client for OpenAI API with fallback handling."""
    
    def __init__(self, api_key: str, console: Any):
        if not OPENAI_AVAILABLE:
            raise RuntimeError("OpenAI package not available. Please install: pip install openai")
            
        self.client = AsyncOpenAI(api_key=api_key)  # type: ignore
        self.BadRequestError = BadRequestError  # type: ignore
        self.console = console

    async def chat(self, messages: List[Dict[str, Any]], 
                  model: Optional[str] = None, 
                  max_tokens: Optional[int] = None, 
                  temperature: Optional[float] = None) -> str:
        """
        Generate chat completion using OpenAI API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            model: Model name to use (defaults to DEFAULT_OPENAI_CHAT_MODEL)
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            Generated text response
        """
        try:
            _model = model or DEFAULT_OPENAI_CHAT_MODEL
            cc = await self.client.chat.completions.create(  # type: ignore
                model=_model, 
                messages=messages,  # type: ignore
                max_tokens=max_tokens, 
                temperature=temperature
            )
            if cc.choices:
                return (cc.choices[0].message.content or "").strip()
            return "[LLM ERROR] No choices returned from API."
            
        except self.BadRequestError as e:  # type: ignore
            # Try a safe fallback model if preview is not available
            try:
                fallback_model = "gpt-4o-mini"
                if (model or DEFAULT_OPENAI_CHAT_MODEL) != fallback_model:
                    cc = await self.client.chat.completions.create(  # type: ignore
                        model=fallback_model, 
                        messages=messages,  # type: ignore
                        max_tokens=max_tokens, 
                        temperature=temperature
                    )
                    if cc.choices:
                        self.console.log(f"[yellow]Fell back to {fallback_model} after BadRequestError for model '{model or DEFAULT_OPENAI_CHAT_MODEL}'.[/yellow]")
                        return (cc.choices[0].message.content or "").strip()
            except Exception:
                pass
            self.console.log(f"[bold red]OpenAI API Error: {e}[/bold red]")
            return f"[LLM ERROR] {e}"

    async def get_logprobs_and_tokens(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[float, List]:
        """Get log probabilities and tokens (not implemented for OpenAI)."""
        return -99.0, []

    async def embedding(self, text: str, 
                       model: Optional[str] = None, 
                       dimensions: Optional[int] = None) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            model: Embedding model name
            dimensions: Number of dimensions (for supported models)
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            _model = model or "text-embedding-3-small"
            if dimensions is None:
                res = await self.client.embeddings.create(input=[text], model=_model)
            else:
                res = await self.client.embeddings.create(
                    input=[text], 
                    model=_model, 
                    dimensions=int(dimensions)
                )
            return res.data[0].embedding
        except Exception as e:
            self.console.log(f"[bold red]OpenAI Embedding Error: {e}[/bold red]")
            return np.zeros(EMBED_DIM).tolist()

    async def batch_embedding(self, texts: List[str], 
                             model: Optional[str] = None, 
                             dimensions: Optional[int] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batch.
        
        Args:
            texts: List of texts to embed
            model: Embedding model name
            dimensions: Number of dimensions (for supported models)
            
        Returns:
            List of embedding vectors
        """
        try:
            _model = model or "text-embedding-3-small"
            if dimensions is None:
                res = await self.client.embeddings.create(input=texts, model=_model)
            else:
                res = await self.client.embeddings.create(
                    input=texts, 
                    model=_model, 
                    dimensions=int(dimensions)
                )
            return [d.embedding for d in res.data]
        except Exception as e:
            self.console.log(f"[bold red]OpenAI Batch Embedding Error: {e}[/bold red]")
            return [np.zeros(EMBED_DIM).tolist() for _ in texts]


class OllamaClient:
    """Async client for Ollama local LLM API."""
    
    def __init__(self, ollama_model: str, console: Any):
        if not OLLAMA_AVAILABLE:
            raise RuntimeError("Python package 'ollama' not installed. Please `pip install ollama`.")
            
        self.client = ollama.AsyncClient()  # type: ignore
        self.model = ollama_model
        self.console = console

    async def chat(self, messages: List[Dict[str, Any]], **kwargs) -> str:
        """
        Generate chat completion using Ollama API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            **kwargs: Additional arguments (ignored for Ollama)
            
        Returns:
            Generated text response
        """
        try:
            res = await self.client.chat(model=self.model, messages=messages)
            return res["message"]["content"].strip()
        except Exception as e:
            self.console.log(f"[bold red]Ollama Chat Error: {e}[/bold red]")
            return f"[LLM ERROR] Could not connect to Ollama or model '{self.model}' not found."

    async def get_logprobs_and_tokens(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[float, List]:
        """Get log probabilities and tokens (not implemented for Ollama)."""
        return -99.0, []

    async def embedding(self, text: str, 
                       model: Optional[str] = None, 
                       dimensions: Optional[int] = None) -> List[float]:
        """
        Generate embedding for a single text using Ollama.
        
        Args:
            text: Text to embed
            model: Model name (defaults to self.model)
            dimensions: Number of dimensions (truncate/pad if specified)
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            res = await self.client.embeddings(model=model or self.model, prompt=text)
            emb = res["embedding"]
            
            if dimensions:
                if len(emb) > dimensions:
                    emb = emb[:dimensions]
                elif len(emb) < dimensions:
                    emb = emb + [0.0] * (dimensions - len(emb))
                    
            return emb
        except Exception as e:
            self.console.log(f"[bold red]Ollama Embedding Error: {e}[/bold red]")
            v = np.random.standard_normal(EMBED_DIM).astype(np.float32)
            v /= (np.linalg.norm(v) + 1e-12)
            return v.tolist()

    async def batch_embedding(self, texts: List[str], 
                             model: Optional[str] = None, 
                             dimensions: Optional[int] = None) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using Ollama.
        
        Args:
            texts: List of texts to embed
            model: Model name (defaults to self.model)
            dimensions: Number of dimensions (truncate/pad if specified)
            
        Returns:
            List of embedding vectors
        """
        try:
            tasks = [self.embedding(t, model, dimensions) for t in texts]
            return await asyncio.gather(*tasks)
        except Exception as e:
            self.console.log(f"[bold red]Ollama Batch Embedding Error: {e}[/bold red]")
            out = []
            for _ in texts:
                v = np.random.standard_normal(EMBED_DIM).astype(np.float32)
                v /= (np.linalg.norm(v) + 1e-12)
                out.append(v.tolist())
            return out


class GeminiClient:
    """Async client for Google Gemini API."""
    
    def __init__(self, api_key: str, model_name: str, console: Any):
        if not GEMINI_AVAILABLE:
            raise RuntimeError("google-generativeai is not installed. Please `pip install google-generativeai`.")
            
        if not api_key:
            raise ValueError("Gemini API key is required.")
            
        # Configure using documented API with fallback handling
        try:
            if hasattr(genai, "configure"):
                genai.configure(api_key=api_key)  # type: ignore
        except Exception:
            pass
            
        try:
            self.model = genai.GenerativeModel(model_name)  # type: ignore
        except Exception:
            # Fallback: some versions use genai.GenerativeModel with model= kw
            try:
                self.model = genai.GenerativeModel(model=model_name)  # type: ignore
            except Exception:
                self.model = None
                
        self.console = console

    async def chat(self, messages: List[Dict[str, Any]], 
                  max_tokens: Optional[int] = None, 
                  temperature: Optional[float] = None, 
                  **kwargs) -> str:
        """
        Generate chat completion using Gemini API.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            **kwargs: Additional arguments
            
        Returns:
            Generated text response
        """
        try:
            # Convert OpenAI format to Gemini format
            gemini_messages = []
            for msg in messages:
                role = "model" if msg["role"] == "assistant" else "user"
                gemini_messages.append({"role": role, "parts": [msg["content"]]})
                
            # Deduplicate consecutive messages with same role
            if len(gemini_messages) > 1:
                deduped = [gemini_messages[0]]
                for i in range(1, len(gemini_messages)):
                    if gemini_messages[i]['role'] != deduped[-1]['role']:
                        deduped.append(gemini_messages[i])
                    else:
                        deduped[-1] = gemini_messages[i]
                gemini_messages = deduped
                
            # Build generation config defensively
            config = None
            try:
                types_mod = getattr(genai, "types", None)
                if types_mod is not None:
                    config = types_mod.GenerationConfig(
                        max_output_tokens=max_tokens, 
                        temperature=temperature
                    )
            except Exception:
                config = None
                
            # Call async or sync version depending on availability
            if hasattr(self.model, "generate_content_async"):
                response = await self.model.generate_content_async(  # type: ignore
                    gemini_messages, generation_config=config
                )
            else:
                response = await asyncio.to_thread(
                    self.model.generate_content,  # type: ignore
                    gemini_messages, 
                    generation_config=config
                )

            # Extract text from response
            text_out = ""
            try:
                candidates = getattr(response, "candidates", []) or []
                chosen = None
                for c in candidates:
                    content = getattr(c, "content", None)
                    parts = getattr(content, "parts", None) if content is not None else None
                    if parts:
                        chosen = c
                        break
                        
                if chosen is not None:
                    parts = chosen.content.parts
                    chunk_list = []
                    for p in parts:
                        t = getattr(p, "text", None)
                        if isinstance(t, str):
                            chunk_list.append(t)
                    text_out = "".join(chunk_list).strip()
                    
                if not text_out:
                    try:
                        text_out = (response.text or "").strip()
                    except Exception:
                        text_out = ""
                        
                if not text_out:
                    try:
                        fr = None
                        if candidates:
                            fr = getattr(candidates[0], "finish_reason", None)
                        self.console.log(f"[bold red]Gemini returned no text. finish_reason={fr}[/bold red]")
                    except Exception:
                        pass
                    return ""
                    
            except Exception as e:
                self.console.log(f"[bold red]Gemini Parse Error: {e}[/bold red]")
                return ""
                
            return text_out
            
        except Exception as e:
            self.console.log(f"[bold red]Gemini Chat Error: {e}[/bold red]")
            return ""

    async def get_logprobs_and_tokens(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[float, List]:
        """Get log probabilities and tokens (not implemented for Gemini)."""
        return -99.0, []

    async def embedding(self, text: str, 
                       model: str = "models/embedding-001", 
                       **kwargs) -> List[float]:
        """
        Generate embedding for a single text using Gemini.
        
        Args:
            text: Text to embed
            model: Embedding model name
            **kwargs: Additional arguments
            
        Returns:
            Embedding vector as list of floats
        """
        try:
            # Use async or sync version depending on availability
            _embed_async = getattr(genai, "embed_content_async", None)
            _embed_sync = getattr(genai, "embed_content", None)
            
            if _embed_async is not None:
                result = await _embed_async(  # type: ignore
                    model=model, 
                    content=text, 
                    task_type="retrieval_document"
                )
            elif _embed_sync is not None:
                result = await asyncio.to_thread(
                    _embed_sync,  # type: ignore
                    model=model, 
                    content=text, 
                    task_type="retrieval_document"
                )
            else:
                return np.zeros(EMBED_DIM).tolist()
                
            # Extract embedding from result
            emb = getattr(result, "embedding", None)
            if emb is None and isinstance(result, dict):
                emb = result.get("embedding")
                
            return emb if emb is not None else np.zeros(EMBED_DIM).tolist()
            
        except Exception as e:
            self.console.log(f"[bold red]Gemini Embedding Error: {e}[/bold red]")
            return np.zeros(EMBED_DIM).tolist()

    async def batch_embedding(self, texts: List[str], 
                             model: str = "models/embedding-001", 
                             **kwargs) -> List[List[float]]:
        """
        Generate embeddings for multiple texts using Gemini.
        
        Args:
            texts: List of texts to embed
            model: Embedding model name
            **kwargs: Additional arguments
            
        Returns:
            List of embedding vectors
        """
        try:
            # Use async or sync version depending on availability
            _embed_async = getattr(genai, "embed_content_async", None)
            _embed_sync = getattr(genai, "embed_content", None)
            
            if _embed_async is not None:
                result = await _embed_async(  # type: ignore
                    model=model, 
                    content=texts, 
                    task_type="retrieval_document"
                )
            elif _embed_sync is not None:
                result = await asyncio.to_thread(
                    _embed_sync,  # type: ignore
                    model=model, 
                    content=texts, 
                    task_type="retrieval_document"
                )
            else:
                return [np.zeros(EMBED_DIM).tolist() for _ in texts]
                
            emb = getattr(result, "embedding", None)
            if emb is None and isinstance(result, dict):
                emb = result.get("embedding")
                
            if isinstance(emb, list):
                return emb
                
            return [np.zeros(EMBED_DIM).tolist() for _ in texts]
            
        except Exception as e:
            self.console.log(f"[bold red]Gemini Batch Embedding Error: {e}[/bold red]")
            return [np.zeros(EMBED_DIM).tolist() for _ in texts]


class AsyncLLMPool:
    """
    Async LLM connection pool for managing concurrent requests.
    Provides load balancing and timeout handling for LLM operations.
    """
    
    def __init__(self, mind_instance: Any, worker_count: int):
        self.mind = mind_instance
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=worker_count * 4)
        self.workers: List[asyncio.Task] = []
        self.worker_count = worker_count
        self.lock = asyncio.Lock()
        self._results: Dict[int, Any] = {}
        self._next_id = 0
        self.running = True
        
        # Configuration from environment
        max_inflight = int(os.getenv('E8_MAX_INFLIGHT', '32'))
        self._sem = asyncio.Semaphore(max_inflight)
        
        try:
            self._timeout = float(os.getenv('E8_LLM_TIMEOUT', '45'))
        except Exception:
            self._timeout = 45.0

    async def _worker(self):
        """Worker coroutine to process LLM requests from the queue."""
        while self.running:
            try:
                item = await self.queue.get()
                if item is None:  # Shutdown signal
                    self.queue.task_done()
                    break
                    
                prompt_id, prompt, args = item

                result = "[LLM UNKNOWN ERROR]"
                try:
                    prompt_key = args.get('_prompt_key', 'ask') if args else 'ask'
                    client_model = getattr(self.mind, 'client_model', 'unknown')
                    
                    self.mind.console.log(
                        f"[LLM POOL] Worker starting task id={prompt_id} "
                        f"key={prompt_key} model={client_model}"
                    )
                    
                    # Call the LLM with timeout
                    result = await asyncio.wait_for(
                        self.mind._async_call_llm_internal(prompt, **(args or {})),
                        timeout=POOL_WORKER_TIMEOUT
                    )
                    
                    self.mind.console.log(f"[LLM POOL] Worker finished task id={prompt_id}")
                    
                except asyncio.TimeoutError:
                    result = f"[LLM TIMEOUT] Task {prompt_id} exceeded {POOL_WORKER_TIMEOUT}s."
                except asyncio.CancelledError:
                    result = "[LLM CANCELLED]"
                    break
                except Exception as e:
                    self.mind.console.log(f"[LLM POOL] Worker error for task {prompt_id}: {e}")
                    result = f"[LLM ERROR] {e}"

                # Store result
                async with self.lock:
                    self._results[prompt_id] = result

                self.queue.task_done()
                
            except Exception as e:
                logging.error(f"LLM Pool worker error: {e}")

    async def start(self):
        """Start the worker pool."""
        self.running = True
        for i in range(self.worker_count):
            worker = asyncio.create_task(self._worker())
            self.workers.append(worker)

    async def stop(self):
        """Stop the worker pool gracefully."""
        self.running = False
        
        # Send shutdown signals
        for _ in range(self.worker_count):
            await self.queue.put(None)
            
        # Wait for workers to finish
        await asyncio.gather(*self.workers, return_exceptions=True)
        self.workers.clear()

    async def submit_prompt(self, prompt: str, **kwargs) -> int:
        """
        Submit a prompt for processing.
        
        Args:
            prompt: The prompt text
            **kwargs: Additional arguments for the LLM call
            
        Returns:
            Task ID for retrieving the result
        """
        async with self.lock:
            prompt_id = self._next_id
            self._next_id += 1

        # Add semaphore acquire/release around queue operations
        await self._sem.acquire()
        try:
            await self.queue.put((prompt_id, prompt, kwargs))
        except:
            self._sem.release()
            raise
            
        return prompt_id

    async def get_result(self, prompt_id: int, timeout: Optional[float] = None) -> str:
        """
        Get the result for a submitted prompt.
        
        Args:
            prompt_id: Task ID from submit_prompt
            timeout: Maximum time to wait for result
            
        Returns:
            LLM response text
        """
        deadline = None
        if timeout is not None:
            deadline = asyncio.get_event_loop().time() + timeout

        while True:
            async with self.lock:
                if prompt_id in self._results:
                    result = self._results.pop(prompt_id)
                    self._sem.release()  # Release semaphore when result is retrieved
                    return result

            if deadline is not None and asyncio.get_event_loop().time() > deadline:
                self._sem.release()  # Release semaphore on timeout
                return f"[LLM TIMEOUT] Result retrieval for task {prompt_id} exceeded {timeout}s."

            await asyncio.sleep(0.1)

    async def submit_and_wait(self, prompt: str, timeout: Optional[float] = None, **kwargs) -> str:
        """
        Submit a prompt and wait for the result.
        
        Args:
            prompt: The prompt text
            timeout: Maximum time to wait for result
            **kwargs: Additional arguments for the LLM call
            
        Returns:
            LLM response text
        """
        if not self.running:
            await self.start()
            
        prompt_id = await self.submit_prompt(prompt, **kwargs)
        return await self.get_result(prompt_id, timeout or self._timeout)

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the queue and pool."""
        return {
            'queue_size': self.queue.qsize(),
            'max_queue_size': self.queue.maxsize,
            'worker_count': len(self.workers),
            'running': self.running,
            'pending_results': len(self._results),
            'next_id': self._next_id,
            'timeout': self._timeout
        }
