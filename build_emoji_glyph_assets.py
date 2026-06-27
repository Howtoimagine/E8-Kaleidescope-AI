#!/usr/bin/env python3
"""
Build emoji and glyph semantic assets for the Kaleidoscope system.

This script generates:
1. emoji_embeddings.npz - Emoji glossary embeddings
2. glyph_semantics.npz - Semantic vectors for 240 E8 glyphs
3. glyph_to_emoji.json - Mapping from glyph indices to top emoji matches

Usage:
    python build_emoji_glyph_assets.py [--provider openai|ollama|gemini|stub]
    
Options:
    --provider: Embedding provider to use (default: stub - random embeddings)
    --dim: Target embedding dimension (default: 512)
    --usage: Include usage-based glyph semantics from logs (default: False)
    --blend: Blend basis and usage semantics (0.0 = basis only, 1.0 = usage only, default: 0.6)
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

import numpy as np

# Import emoji_semantics module
from emoji_semantics import (
    EMOJI_CANDIDATES,
    DEFAULT_EMOJI_EMBED_PATH,
    DEFAULT_GLYPH_SEMANTIC_PATH,
    DEFAULT_GLYPH_TO_EMOJI_PATH,
    build_emoji_embedding_table,
    save_emoji_embeddings,
    build_glyph_semantics_from_basis,
    load_glyph_usage_from_logs,
    build_glyph_semantics_from_usage,
    blend_semantic_sources,
    save_glyph_semantics,
    build_glyph_to_emoji_map,
    save_glyph_to_emoji_map,
)


class StubEmbedder:
    """Stub embedder using random vectors for testing."""
    
    def __init__(self, dim: int = 512):
        self.dim = dim
        self.rng = np.random.RandomState(42)
    
    def embed(self, text: str) -> np.ndarray:
        """Generate deterministic random embedding based on text hash."""
        # Use text hash as seed for determinism
        seed = hash(text) % (2**31)
        local_rng = np.random.RandomState(seed)
        vec = local_rng.randn(self.dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec


class AsyncOpenAIEmbedder:
    """OpenAI embedder using text-embedding-3-small."""
    
    def __init__(self, api_key: str, model: str = "text-embedding-3-small", dim: int = 512):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(api_key=api_key)
        self.model = model
        self.dim = dim
    
    async def embed_async(self, text: str) -> np.ndarray:
        """Async embed single text."""
        response = await self.client.embeddings.create(
            model=self.model,
            input=text,
            dimensions=self.dim
        )
        return np.array(response.data[0].embedding, dtype=np.float32)
    
    def embed(self, text: str) -> np.ndarray:
        """Sync wrapper for embed."""
        return asyncio.run(self.embed_async(text))


class OllamaEmbedder:
    """Ollama embedder using nomic-embed-text."""
    
    def __init__(self, model: str = "nomic-embed-text", dim: int = 768):
        import ollama
        self.client = ollama
        self.model = model
        self.dim = dim
    
    def embed(self, text: str) -> np.ndarray:
        """Embed single text."""
        response = self.client.embeddings(model=self.model, prompt=text)
        vec = np.array(response['embedding'], dtype=np.float32)
        # Ollama may not support dimension parameter, so we truncate/pad
        if len(vec) > self.dim:
            vec = vec[:self.dim]
        elif len(vec) < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[:len(vec)] = vec
            vec = padded
        return vec


class GeminiEmbedder:
    """Google Gemini embedder."""
    
    def __init__(self, api_key: str, model: str = "models/embedding-001", dim: int = 768):
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = model
        self.dim = dim
    
    def embed(self, text: str) -> np.ndarray:
        """Embed single text."""
        import google.generativeai as genai
        result = genai.embed_content(model=self.model, content=text, task_type="retrieval_document")
        vec = np.array(result['embedding'], dtype=np.float32)
        # Truncate or pad to target dimension
        if len(vec) > self.dim:
            vec = vec[:self.dim]
        elif len(vec) < self.dim:
            padded = np.zeros(self.dim, dtype=np.float32)
            padded[:len(vec)] = vec
            vec = padded
        return vec


def create_embedder(provider: str, dim: int):
    """Create appropriate embedder based on provider."""
    if provider == "stub":
        print(f"📦 Using stub embedder (random vectors, dim={dim})")
        return StubEmbedder(dim=dim)
    
    elif provider == "openai":
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable not set")
        print(f"🔑 Using OpenAI embedder (text-embedding-3-small, dim={dim})")
        return AsyncOpenAIEmbedder(api_key=api_key, dim=dim)
    
    elif provider == "ollama":
        print(f"🦙 Using Ollama embedder (nomic-embed-text, dim={dim})")
        return OllamaEmbedder(dim=dim)
    
    elif provider == "gemini":
        api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY environment variable not set")
        print(f"💎 Using Gemini embedder (embedding-001, dim={dim})")
        return GeminiEmbedder(api_key=api_key, dim=dim)
    
    else:
        raise ValueError(f"Unknown provider: {provider}")


def generate_e8_basis(dim: int = 512) -> np.ndarray:
    """Generate E8 root basis for 240 glyphs.
    
    Returns a 240 x dim matrix where each row is a glyph basis vector.
    For simplicity, we generate orthonormal random vectors.
    In a real system, these would be derived from the E8 root system.
    """
    print(f"🔮 Generating E8 basis (240 glyphs x {dim} dims)")
    
    # Use fixed seed for reproducibility
    rng = np.random.RandomState(42)
    
    # Generate 240 random vectors
    basis = rng.randn(240, dim).astype(np.float32)
    
    # Normalize each row
    norms = np.linalg.norm(basis, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    basis = basis / norms
    
    return basis


def main():
    parser = argparse.ArgumentParser(description="Build emoji and glyph semantic assets")
    parser.add_argument("--provider", choices=["stub", "openai", "ollama", "gemini"], 
                       default="stub", help="Embedding provider")
    parser.add_argument("--dim", type=int, default=512, help="Target embedding dimension")
    parser.add_argument("--usage", action="store_true", help="Include usage-based semantics from logs")
    parser.add_argument("--blend", type=float, default=0.6, help="Blend weight for usage semantics (0-1)")
    parser.add_argument("--topk", type=int, default=3, help="Top K emoji matches per glyph")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🌈 Kaleidoscope Emoji & Glyph Asset Builder")
    print("=" * 60)
    
    # Create embedder
    try:
        embedder = create_embedder(args.provider, args.dim)
    except Exception as e:
        print(f"❌ Failed to create embedder: {e}")
        return 1
    
    # Step 1: Build emoji embeddings
    print(f"\n📝 Step 1: Building emoji embeddings ({len(EMOJI_CANDIDATES)} emojis)...")
    try:
        emoji_table = build_emoji_embedding_table(
            embed_fn=embedder.embed,
            candidates=EMOJI_CANDIDATES,
            target_dim=args.dim,
            logger=lambda msg: print(f"  {msg}")
        )
        print(f"✅ Built {len(emoji_table)} emoji embeddings")
        
        # Save emoji embeddings
        emoji_path = save_emoji_embeddings(emoji_table, DEFAULT_EMOJI_EMBED_PATH)
        print(f"💾 Saved emoji embeddings to: {emoji_path}")
    except Exception as e:
        print(f"❌ Failed to build emoji embeddings: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 2: Build glyph semantics
    print(f"\n🔮 Step 2: Building glyph semantics (240 glyphs)...")
    try:
        # Generate E8 basis
        e8_basis = generate_e8_basis(dim=args.dim)
        
        # Option A: Build from basis
        basis_semantics = build_glyph_semantics_from_basis(
            basis_matrix=e8_basis,
            adapter=None  # No adapter for now
        )
        print(f"✅ Built basis semantics: {basis_semantics.shape}")
        
        # Option B: Build from usage (if requested)
        usage_semantics = None
        if args.usage:
            print("📊 Loading glyph usage from logs...")
            # Look for NDJSON logs in common locations
            log_paths = [
                "insights.ndjson",
                "metrics.ndjson",
                "subconscious_narrative.ndjson",
                "runtime/console.ndjson"
            ]
            existing_logs = [p for p in log_paths if Path(p).exists()]
            
            if existing_logs:
                print(f"📂 Found {len(existing_logs)} log files")
                usage_dict = load_glyph_usage_from_logs(existing_logs)
                print(f"📈 Loaded usage for {len(usage_dict)} glyphs")
                
                usage_semantics = build_glyph_semantics_from_usage(
                    embed_fn=embedder.embed,
                    usage_dict=usage_dict,
                    glyph_count=240,
                    target_dim=args.dim,
                    fallback_vectors=basis_semantics
                )
                print(f"✅ Built usage semantics: {usage_semantics.shape}")
            else:
                print("⚠️  No log files found, skipping usage-based semantics")
        
        # Blend if both are available
        if usage_semantics is not None:
            print(f"🎨 Blending basis and usage semantics (weight={args.blend})...")
            glyph_semantics = blend_semantic_sources(
                basis_semantics=basis_semantics,
                usage_semantics=usage_semantics,
                usage_weight=args.blend
            )
            print(f"✅ Blended semantics: {glyph_semantics.shape}")
        else:
            glyph_semantics = basis_semantics
        
        # Save glyph semantics
        glyph_path = save_glyph_semantics(glyph_semantics, DEFAULT_GLYPH_SEMANTIC_PATH)
        print(f"💾 Saved glyph semantics to: {glyph_path}")
    except Exception as e:
        print(f"❌ Failed to build glyph semantics: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Step 3: Build glyph-to-emoji mapping
    print(f"\n🗺️  Step 3: Building glyph-to-emoji mapping (top-{args.topk})...")
    try:
        # Extract emoji list and vectors from emoji_table
        emoji_list = list(emoji_table.keys())
        emoji_vecs = np.stack([emoji_table[e] for e in emoji_list], axis=0)
        
        # Build mapping
        glyph_to_emoji = build_glyph_to_emoji_map(
            glyph_vecs=glyph_semantics,
            emoji_vecs=emoji_vecs,
            emoji_list=emoji_list,
            topk=args.topk
        )
        print(f"✅ Built mapping for {len(glyph_to_emoji)} glyphs")
        
        # Save mapping
        mapping_path = save_glyph_to_emoji_map(glyph_to_emoji, DEFAULT_GLYPH_TO_EMOJI_PATH)
        print(f"💾 Saved glyph-to-emoji mapping to: {mapping_path}")
    except Exception as e:
        print(f"❌ Failed to build glyph-to-emoji mapping: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Summary
    print("\n" + "=" * 60)
    print("✨ Build Complete!")
    print("=" * 60)
    print(f"📄 Generated files:")
    print(f"  1. {DEFAULT_EMOJI_EMBED_PATH}")
    print(f"  2. {DEFAULT_GLYPH_SEMANTIC_PATH}")
    print(f"  3. {DEFAULT_GLYPH_TO_EMOJI_PATH}")
    print()
    print("🚀 Ready to start the server!")
    print("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
