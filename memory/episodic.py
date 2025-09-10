"""
E8Mind Episodic Memory System - Extracted from M18.7

This module provides episodic memory capabilities with graph-based storage,
semantic embedding, and temporal reasoning.
"""

import asyncio
import json
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
import traceback

# Safe JSON utilities
def safe_json_read(file_path: str, default=None):
    """Safely read JSON from file with fallback."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default if default is not None else {}

def safe_json_write(file_path: str, data: Any):
    """Safely write JSON to file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        return True
    except Exception:
        return False


class NoveltyScorer:
    """
    Computes novelty scores for experiences based on similarity to past memories.
    Uses a simple distance-based approach with temporal decay.
    """
    
    def __init__(self, base_threshold=0.7, decay_factor=0.95):
        self.base_threshold = base_threshold
        self.decay_factor = decay_factor
        self.memory_embeddings = []
        self.memory_timestamps = []
        
    def add_memory(self, embedding: np.ndarray, timestamp: Optional[float] = None):
        """Add a memory embedding for future novelty comparisons."""
        if timestamp is None:
            timestamp = datetime.now().timestamp()
            
        self.memory_embeddings.append(embedding)
        self.memory_timestamps.append(timestamp)
        
    def score_novelty(self, new_embedding: np.ndarray) -> float:
        """
        Score the novelty of a new experience.
        Returns 1.0 for completely novel, 0.0 for completely familiar.
        """
        if not self.memory_embeddings:
            return 1.0
            
        current_time = datetime.now().timestamp()
        max_similarity = 0.0
        
        for i, (memory_emb, timestamp) in enumerate(zip(self.memory_embeddings, self.memory_timestamps)):
            # Compute similarity
            similarity = np.dot(new_embedding, memory_emb) / (
                np.linalg.norm(new_embedding) * np.linalg.norm(memory_emb) + 1e-8
            )
            
            # Apply temporal decay
            time_diff = current_time - timestamp
            decay = self.decay_factor ** (time_diff / 86400)  # Daily decay
            weighted_similarity = similarity * decay
            
            max_similarity = max(max_similarity, weighted_similarity)
            
        # Convert similarity to novelty score
        novelty = max(0.0, 1.0 - max_similarity)
        return float(novelty)
        
    def update_threshold(self, new_threshold: float):
        """Update the novelty threshold."""
        self.base_threshold = max(0.0, min(1.0, new_threshold))


class EpisodicMemory:
    """
    Episodic memory system for storing and retrieving personal experiences.
    Combines semantic embeddings with temporal and contextual metadata.
    """
    
    def __init__(self, max_episodes=10000, embedding_dim=768):
        self.max_episodes = max_episodes
        self.embedding_dim = embedding_dim
        self.episodes = []
        self.embeddings = np.zeros((max_episodes, embedding_dim), dtype=np.float32)
        self.episode_count = 0
        self.novelty_scorer = NoveltyScorer()
        
        # Metadata tracking
        self.last_access_times = {}
        self.access_counts = {}
        
    def store_episode(self, 
                     content: str, 
                     embedding: np.ndarray, 
                     context: Optional[Dict[str, Any]] = None,
                     emotional_state: Optional[Dict[str, float]] = None,
                     importance: float = 0.5) -> int:
        """
        Store a new episodic memory.
        
        Args:
            content: The textual content of the episode
            embedding: Semantic embedding of the episode
            context: Additional contextual information
            emotional_state: Emotional context at time of encoding
            importance: Importance score (0-1)
            
        Returns:
            Episode ID
        """
        timestamp = datetime.now().timestamp()
        
        # Compute novelty score
        novelty = self.novelty_scorer.score_novelty(embedding)
        
        # Create episode record
        episode = {
            'id': self.episode_count,
            'content': content,
            'timestamp': timestamp,
            'context': context or {},
            'emotional_state': emotional_state or {},
            'importance': importance,
            'novelty': novelty,
            'access_count': 0,
            'last_accessed': timestamp
        }
        
        # Store episode and embedding
        if self.episode_count < self.max_episodes:
            self.episodes.append(episode)
            self.embeddings[self.episode_count] = embedding
        else:
            # Replace least important episode
            least_important_idx = self._find_least_important()
            self.episodes[least_important_idx] = episode
            self.embeddings[least_important_idx] = embedding
            episode['id'] = least_important_idx
            
        # Update novelty scorer
        self.novelty_scorer.add_memory(embedding, timestamp)
        
        episode_id = episode['id']
        self.episode_count = min(self.episode_count + 1, self.max_episodes)
        
        return episode_id
        
    def retrieve_episodes(self, 
                         query_embedding: np.ndarray, 
                         k: int = 5,
                         time_window: Optional[Tuple[float, float]] = None,
                         importance_threshold: float = 0.0) -> List[Dict[str, Any]]:
        """
        Retrieve k most similar episodes to the query.
        
        Args:
            query_embedding: Query embedding to match against
            k: Number of episodes to retrieve
            time_window: Optional (start_time, end_time) filter
            importance_threshold: Minimum importance score
            
        Returns:
            List of episode dictionaries with similarity scores
        """
        if not self.episodes:
            return []
            
        # Compute similarities
        valid_embeddings = self.embeddings[:len(self.episodes)]
        similarities = np.dot(valid_embeddings, query_embedding) / (
            np.linalg.norm(valid_embeddings, axis=1) * np.linalg.norm(query_embedding) + 1e-8
        )
        
        # Filter episodes
        candidates = []
        for i, (episode, similarity) in enumerate(zip(self.episodes, similarities)):
            # Apply filters
            if importance_threshold > 0 and episode['importance'] < importance_threshold:
                continue
                
            if time_window:
                start_time, end_time = time_window
                if not (start_time <= episode['timestamp'] <= end_time):
                    continue
                    
            candidates.append({
                'episode': episode,
                'similarity': float(similarity),
                'index': i
            })
            
        # Sort by similarity and take top k
        candidates.sort(key=lambda x: x['similarity'], reverse=True)
        results = candidates[:k]
        
        # Update access statistics
        current_time = datetime.now().timestamp()
        for result in results:
            episode_id = result['episode']['id']
            self.access_counts[episode_id] = self.access_counts.get(episode_id, 0) + 1
            self.last_access_times[episode_id] = current_time
            result['episode']['access_count'] = self.access_counts[episode_id]
            result['episode']['last_accessed'] = current_time
            
        return results
        
    def get_episode_by_id(self, episode_id: int) -> Optional[Dict[str, Any]]:
        """Retrieve a specific episode by ID."""
        for episode in self.episodes:
            if episode['id'] == episode_id:
                # Update access stats
                current_time = datetime.now().timestamp()
                self.access_counts[episode_id] = self.access_counts.get(episode_id, 0) + 1
                self.last_access_times[episode_id] = current_time
                episode['access_count'] = self.access_counts[episode_id]
                episode['last_accessed'] = current_time
                return episode
        return None
        
    def update_importance(self, episode_id: int, new_importance: float):
        """Update the importance score of an episode."""
        for episode in self.episodes:
            if episode['id'] == episode_id:
                episode['importance'] = max(0.0, min(1.0, new_importance))
                break
                
    def get_recent_episodes(self, days: int = 7, k: int = 10) -> List[Dict[str, Any]]:
        """Get recent episodes within the specified time window."""
        current_time = datetime.now().timestamp()
        cutoff_time = current_time - (days * 24 * 3600)
        
        recent = [ep for ep in self.episodes if ep['timestamp'] >= cutoff_time]
        recent.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return recent[:k]
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics."""
        if not self.episodes:
            return {'total_episodes': 0}
            
        importances = [ep['importance'] for ep in self.episodes]
        novelties = [ep['novelty'] for ep in self.episodes]
        
        return {
            'total_episodes': len(self.episodes),
            'avg_importance': np.mean(importances),
            'avg_novelty': np.mean(novelties),
            'most_accessed': max(self.access_counts.values()) if self.access_counts else 0,
            'memory_utilization': len(self.episodes) / self.max_episodes
        }
        
    def _find_least_important(self) -> int:
        """Find the index of the least important episode for replacement."""
        if not self.episodes:
            return 0
            
        # Score episodes by importance, novelty, and recency
        current_time = datetime.now().timestamp()
        scores = []
        
        for i, episode in enumerate(self.episodes):
            age_days = (current_time - episode['timestamp']) / 86400
            recency_factor = 1.0 / (1.0 + age_days * 0.1)  # Decay with age
            
            access_factor = 1.0 / (1.0 + self.access_counts.get(episode['id'], 0) * 0.1)
            
            # Combined score (lower is more likely to be replaced)
            score = (episode['importance'] * episode['novelty'] * 
                    recency_factor * access_factor)
            scores.append(score)
            
        return int(np.argmin(scores))
        
    def save_to_file(self, filepath: str) -> bool:
        """Save episodic memory to file."""
        try:
            data = {
                'episodes': self.episodes,
                'embeddings': self.embeddings[:len(self.episodes)].tolist(),
                'episode_count': self.episode_count,
                'access_counts': self.access_counts,
                'last_access_times': self.last_access_times,
                'max_episodes': self.max_episodes,
                'embedding_dim': self.embedding_dim
            }
            return safe_json_write(filepath, data)
        except Exception:
            return False
            
    def load_from_file(self, filepath: str) -> bool:
        """Load episodic memory from file."""
        try:
            data = safe_json_read(filepath)
            if not data:
                return False
                
            self.episodes = data.get('episodes', [])
            embeddings_list = data.get('embeddings', [])
            if embeddings_list:
                self.embeddings[:len(embeddings_list)] = np.array(embeddings_list)
            self.episode_count = data.get('episode_count', 0)
            self.access_counts = data.get('access_counts', {})
            self.last_access_times = data.get('last_access_times', {})
            
            return True
        except Exception:
            return False
