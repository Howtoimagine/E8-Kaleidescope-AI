"""
E8Mind Agent System - Extracted from M18.7

This module provides various specialized agents for different cognitive functions,
including insight generation, meta-arbitration, and multi-agent coordination.
"""

import asyncio
import json
import logging
import numpy as np
from typing import Any, Dict, List, Optional, Tuple, Union, Callable
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


class BaseAgentAdapter:
    """Base adapter for different agent interfaces."""
    
    def __init__(self, name: str):
        self.name = name
        self.available = True
        
    async def get_action(self, state: Any) -> Any:
        """Get action from agent given current state."""
        return None
        
    async def update(self, experience: Dict[str, Any]):
        """Update agent with experience."""
        pass


class ActionCandidateSampler:
    """Samples action candidates for exploration and planning."""
    
    def __init__(self, action_dim: int, num_candidates: int = 10):
        self.action_dim = action_dim
        self.num_candidates = num_candidates
        
    def sample_candidates(self, context: Optional[Dict[str, Any]] = None) -> List[np.ndarray]:
        """Sample action candidates."""
        candidates = []
        for _ in range(self.num_candidates):
            # Simple random sampling - can be made more sophisticated
            action = np.random.randn(self.action_dim)
            candidates.append(action)
        return candidates
        
    def sample_around_action(self, base_action: np.ndarray, noise_scale: float = 0.1) -> List[np.ndarray]:
        """Sample candidates around a base action."""
        candidates = []
        for _ in range(self.num_candidates):
            noise = np.random.normal(0, noise_scale, self.action_dim)
            candidate = base_action + noise
            candidates.append(candidate)
        return candidates


class NoveltyAgent(BaseAgentAdapter):
    """Agent that prioritizes novel and exploratory actions."""
    
    def __init__(self, name: str = "NoveltyAgent"):
        super().__init__(name)
        self.exploration_history = []
        self.novelty_threshold = 0.5
        
    async def get_action(self, state: Any) -> Any:
        """Get action that maximizes novelty."""
        # Simple novelty-seeking behavior
        action = np.random.randn(4)  # 4D action space by default
        self.exploration_history.append(action.copy())
        return action
        
    async def update(self, experience: Dict[str, Any]):
        """Update novelty estimates."""
        if 'novelty_score' in experience:
            # Adjust exploration based on novelty feedback
            score = experience['novelty_score']
            if score < self.novelty_threshold:
                self.novelty_threshold = max(0.1, self.novelty_threshold * 0.95)


class StabilityAgent(BaseAgentAdapter):
    """Agent that prioritizes stable and reliable actions."""
    
    def __init__(self, name: str = "StabilityAgent"):
        super().__init__(name)
        self.stable_actions = []
        self.reliability_scores = {}
        
    async def get_action(self, state: Any) -> Any:
        """Get action that maximizes stability."""
        if self.stable_actions:
            # Choose from known stable actions
            best_action = max(self.stable_actions, 
                            key=lambda a: self.reliability_scores.get(str(a), 0.0))
            return best_action
        else:
            # Conservative random action
            return np.random.randn(4) * 0.5  # Smaller magnitude for stability
            
    async def update(self, experience: Dict[str, Any]):
        """Update stability estimates."""
        if 'action' in experience and 'reward' in experience:
            action = experience['action']
            reward = experience['reward']
            action_key = str(action)
            
            # Update reliability score
            if action_key not in self.reliability_scores:
                self.reliability_scores[action_key] = 0.0
                
            # Exponential moving average
            self.reliability_scores[action_key] = (
                0.9 * self.reliability_scores[action_key] + 0.1 * reward
            )
            
            # Add to stable actions if reliable enough
            if (self.reliability_scores[action_key] > 0.7 and 
                action not in self.stable_actions):
                self.stable_actions.append(action)


class SynthesisAgent(BaseAgentAdapter):
    """Agent that synthesizes insights from multiple sources."""
    
    def __init__(self, name: str = "SynthesisAgent"):
        super().__init__(name)
        self.synthesis_history = []
        self.source_weights = {}
        
    async def get_action(self, state: Any) -> Any:
        """Synthesize action from multiple considerations."""
        # Combine different action sources
        components = []
        
        # Add some exploration component
        exploration = np.random.randn(4) * 0.3
        components.append(('exploration', exploration, 0.3))
        
        # Add some stability component
        stability = np.random.randn(4) * 0.1  # Small stable action
        components.append(('stability', stability, 0.4))
        
        # Add some goal-directed component
        goal_directed = np.random.randn(4) * 0.5
        components.append(('goal', goal_directed, 0.3))
        
        # Weighted synthesis
        final_action = np.zeros(4)
        total_weight = 0.0
        
        for source, action, weight in components:
            source_weight = self.source_weights.get(source, 1.0)
            effective_weight = weight * source_weight
            final_action += action * effective_weight
            total_weight += effective_weight
            
        if total_weight > 0:
            final_action /= total_weight
            
        self.synthesis_history.append({
            'components': components,
            'final_action': final_action.copy(),
            'timestamp': datetime.now().timestamp()
        })
        
        return final_action
        
    async def update(self, experience: Dict[str, Any]):
        """Update synthesis weights based on outcomes."""
        if 'reward' in experience and self.synthesis_history:
            reward = experience['reward']
            last_synthesis = self.synthesis_history[-1]
            
            # Update source weights based on contribution to reward
            for source, action, weight in last_synthesis['components']:
                if source not in self.source_weights:
                    self.source_weights[source] = 1.0
                    
                # Simple reward-based update
                self.source_weights[source] = (
                    0.9 * self.source_weights[source] + 0.1 * max(0, reward)
                )


class MetaArbiter:
    """Meta-level coordinator that manages multiple agents and strategies."""
    
    def __init__(self):
        self.agents = {}
        self.agent_performance = {}
        self.selection_history = []
        self.coordination_strategy = "weighted_voting"
        
    def register_agent(self, agent: BaseAgentAdapter, initial_weight: float = 1.0):
        """Register an agent with the arbiter."""
        self.agents[agent.name] = agent
        self.agent_performance[agent.name] = {
            'weight': initial_weight,
            'success_rate': 0.5,
            'recent_rewards': [],
            'selection_count': 0
        }
        
    async def coordinate_action(self, state: Any, context: Optional[Dict[str, Any]] = None) -> Any:
        """Coordinate actions from multiple agents."""
        if not self.agents:
            return np.random.randn(4)  # Default action
            
        agent_actions = {}
        agent_confidences = {}
        
        # Get actions from all available agents
        for name, agent in self.agents.items():
            if agent.available:
                try:
                    action = await agent.get_action(state)
                    agent_actions[name] = action
                    
                    # Compute confidence based on past performance
                    perf = self.agent_performance[name]
                    confidence = perf['weight'] * perf['success_rate']
                    agent_confidences[name] = confidence
                    
                except Exception as e:
                    logging.warning(f"Agent {name} failed to provide action: {e}")
                    
        if not agent_actions:
            return np.random.randn(4)
            
        # Coordinate based on strategy
        if self.coordination_strategy == "weighted_voting":
            final_action = await self._weighted_voting(agent_actions, agent_confidences)
        elif self.coordination_strategy == "best_agent":
            final_action = await self._best_agent_selection(agent_actions, agent_confidences)
        else:
            # Default to averaging
            actions_array = np.array(list(agent_actions.values()))
            final_action = np.mean(actions_array, axis=0)
            
        # Record coordination decision
        self.selection_history.append({
            'timestamp': datetime.now().timestamp(),
            'agent_actions': agent_actions,
            'agent_confidences': agent_confidences,
            'final_action': final_action.copy() if hasattr(final_action, 'copy') else final_action,
            'strategy': self.coordination_strategy
        })
        
        return final_action
        
    async def _weighted_voting(self, agent_actions: Dict[str, Any], 
                              agent_confidences: Dict[str, float]) -> Any:
        """Coordinate using weighted voting."""
        if not agent_actions:
            return np.random.randn(4)
            
        # Normalize confidences
        total_confidence = sum(agent_confidences.values())
        if total_confidence == 0:
            # Equal weights if no confidence info
            weights = {name: 1.0/len(agent_actions) for name in agent_actions}
        else:
            weights = {name: conf/total_confidence 
                      for name, conf in agent_confidences.items()}
            
        # Weighted combination
        final_action = None
        for name, action in agent_actions.items():
            weight = weights[name]
            
            if final_action is None:
                if isinstance(action, np.ndarray):
                    final_action = action * weight
                else:
                    final_action = action  # For non-array actions
            else:
                if isinstance(action, np.ndarray):
                    final_action += action * weight
                    
        return final_action
        
    async def _best_agent_selection(self, agent_actions: Dict[str, Any], 
                                   agent_confidences: Dict[str, float]) -> Any:
        """Select action from the most confident agent."""
        if not agent_confidences:
            return list(agent_actions.values())[0]
            
        best_agent = max(agent_confidences.keys(), 
                        key=lambda x: agent_confidences[x])
        
        # Update selection count
        self.agent_performance[best_agent]['selection_count'] += 1
        
        return agent_actions[best_agent]
        
    async def update_agents(self, experience: Dict[str, Any]):
        """Update all agents with experience and adjust meta-parameters."""
        reward = experience.get('reward', 0.0)
        
        # Update individual agents
        for agent in self.agents.values():
            try:
                await agent.update(experience)
            except Exception as e:
                logging.warning(f"Failed to update agent {agent.name}: {e}")
                
        # Update meta-level performance tracking
        if self.selection_history:
            last_decision = self.selection_history[-1]
            
            # Update performance metrics for contributing agents
            for name in last_decision['agent_actions'].keys():
                perf = self.agent_performance[name]
                perf['recent_rewards'].append(reward)
                
                # Keep only recent rewards (last 10)
                if len(perf['recent_rewards']) > 10:
                    perf['recent_rewards'] = perf['recent_rewards'][-10:]
                    
                # Update success rate
                positive_rewards = sum(1 for r in perf['recent_rewards'] if r > 0)
                perf['success_rate'] = positive_rewards / len(perf['recent_rewards'])
                
                # Update weight based on performance
                if perf['recent_rewards']:
                    avg_reward = np.mean(perf['recent_rewards'])
                    perf['weight'] = max(0.1, perf['weight'] * (1.0 + avg_reward * 0.1))
                    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get statistics about coordination performance."""
        stats = {
            'total_agents': len(self.agents),
            'coordination_strategy': self.coordination_strategy,
            'total_decisions': len(self.selection_history),
            'agent_performance': {}
        }
        
        for name, perf in self.agent_performance.items():
            stats['agent_performance'][name] = {
                'weight': perf['weight'],
                'success_rate': perf['success_rate'],
                'selection_count': perf['selection_count'],
                'avg_recent_reward': np.mean(perf['recent_rewards']) if perf['recent_rewards'] else 0.0
            }
            
        return stats
        
    def set_coordination_strategy(self, strategy: str):
        """Set the coordination strategy."""
        valid_strategies = ["weighted_voting", "best_agent", "averaging"]
        if strategy in valid_strategies:
            self.coordination_strategy = strategy
        else:
            logging.warning(f"Invalid strategy {strategy}, keeping {self.coordination_strategy}")


class InsightAgent:
    """
    Agent specialized in generating insights from experiences and patterns.
    Focuses on meta-cognitive awareness and strategic thinking.
    """
    
    def __init__(self, name: str = "InsightAgent"):
        self.name = name
        self.insights_generated = []
        self.pattern_memory = []
        self.meta_observations = []
        
    async def analyze_experience(self, experience: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze an experience to extract insights and patterns.
        
        Args:
            experience: Dictionary containing experience data
            
        Returns:
            Dictionary containing insights and meta-observations
        """
        insights = {
            'timestamp': datetime.now().timestamp(),
            'experience_summary': self._summarize_experience(experience),
            'patterns_detected': [],
            'strategic_recommendations': [],
            'meta_observations': [],
            'confidence': 0.5
        }
        
        # Pattern detection
        patterns = await self._detect_patterns(experience)
        insights['patterns_detected'] = patterns
        
        # Strategic analysis
        strategies = await self._generate_strategic_recommendations(experience, patterns)
        insights['strategic_recommendations'] = strategies
        
        # Meta-cognitive observations
        meta_obs = await self._generate_meta_observations(experience)
        insights['meta_observations'] = meta_obs
        
        # Store insight
        self.insights_generated.append(insights)
        
        return insights
        
    def _summarize_experience(self, experience: Dict[str, Any]) -> str:
        """Generate a textual summary of the experience."""
        summary_parts = []
        
        if 'state' in experience:
            summary_parts.append(f"State context provided")
            
        if 'action' in experience:
            summary_parts.append(f"Action taken")
            
        if 'reward' in experience:
            reward = experience['reward']
            if reward > 0:
                summary_parts.append(f"Positive outcome (reward: {reward:.3f})")
            elif reward < 0:
                summary_parts.append(f"Negative outcome (reward: {reward:.3f})")
            else:
                summary_parts.append("Neutral outcome")
                
        if 'novelty_score' in experience:
            novelty = experience['novelty_score']
            if novelty > 0.7:
                summary_parts.append("High novelty experience")
            elif novelty < 0.3:
                summary_parts.append("Familiar experience")
                
        return "; ".join(summary_parts) if summary_parts else "Basic experience recorded"
        
    async def _detect_patterns(self, experience: Dict[str, Any]) -> List[str]:
        """Detect patterns in the current experience relative to history."""
        patterns = []
        
        # Add experience to pattern memory
        self.pattern_memory.append(experience)
        
        # Keep only recent experiences for pattern detection
        if len(self.pattern_memory) > 100:
            self.pattern_memory = self.pattern_memory[-100:]
            
        # Simple pattern detection
        if len(self.pattern_memory) >= 3:
            recent_rewards = [exp.get('reward', 0) for exp in self.pattern_memory[-5:]]
            
            if all(r > 0 for r in recent_rewards[-3:]):
                patterns.append("Positive trend detected")
            elif all(r < 0 for r in recent_rewards[-3:]):
                patterns.append("Negative trend detected")
                
            # Novelty patterns
            recent_novelty = [exp.get('novelty_score', 0.5) for exp in self.pattern_memory[-5:]]
            avg_novelty = np.mean(recent_novelty)
            
            if avg_novelty > 0.7:
                patterns.append("High exploration phase")
            elif avg_novelty < 0.3:
                patterns.append("Exploitation phase")
                
        return patterns
        
    async def _generate_strategic_recommendations(self, 
                                                experience: Dict[str, Any], 
                                                patterns: List[str]) -> List[str]:
        """Generate strategic recommendations based on experience and patterns."""
        recommendations = []
        
        # Reward-based recommendations
        reward = experience.get('reward', 0)
        if reward > 0.5:
            recommendations.append("Continue current strategy - high reward achieved")
        elif reward < -0.5:
            recommendations.append("Consider strategy adjustment - negative outcomes")
            
        # Pattern-based recommendations
        for pattern in patterns:
            if "Positive trend" in pattern:
                recommendations.append("Maintain current approach - positive momentum")
            elif "Negative trend" in pattern:
                recommendations.append("Pivot strategy - breaking negative cycle needed")
            elif "High exploration" in pattern:
                recommendations.append("Balance exploration with exploitation")
            elif "Exploitation" in pattern:
                recommendations.append("Consider increasing exploration for novelty")
                
        # Novelty-based recommendations
        novelty = experience.get('novelty_score', 0.5)
        if novelty < 0.2:
            recommendations.append("Seek more novel experiences to avoid stagnation")
        elif novelty > 0.9:
            recommendations.append("Consider consolidating recent novel experiences")
            
        return recommendations
        
    async def _generate_meta_observations(self, experience: Dict[str, Any]) -> List[str]:
        """Generate meta-cognitive observations about thinking and decision-making."""
        observations = []
        
        # Analyze decision quality
        if 'action' in experience and 'reward' in experience:
            reward = experience['reward']
            if reward > 0:
                observations.append("Decision-making process yielded positive results")
            else:
                observations.append("Decision-making process needs refinement")
                
        # Analyze exploration-exploitation balance
        novelty = experience.get('novelty_score', 0.5)
        if novelty > 0.8:
            observations.append("High exploration - good for discovery but may need focus")
        elif novelty < 0.2:
            observations.append("Low exploration - efficient but may miss opportunities")
            
        # Memory and learning observations
        if len(self.insights_generated) > 0:
            recent_insights = self.insights_generated[-5:]
            confidence_trend = [ins['confidence'] for ins in recent_insights]
            
            if len(confidence_trend) >= 3:
                if np.mean(confidence_trend[-3:]) > np.mean(confidence_trend[:-3]):
                    observations.append("Insight quality improving - learning is effective")
                else:
                    observations.append("Insight quality declining - may need new approaches")
                    
        self.meta_observations.extend(observations)
        
        return observations
        
    def get_recent_insights(self, num_insights: int = 5) -> List[Dict[str, Any]]:
        """Get the most recent insights generated."""
        return self.insights_generated[-num_insights:] if self.insights_generated else []
        
    def get_insight_summary(self) -> Dict[str, Any]:
        """Get a summary of all insights and patterns."""
        if not self.insights_generated:
            return {'total_insights': 0}
            
        all_patterns = []
        all_recommendations = []
        all_meta_obs = []
        
        for insight in self.insights_generated:
            all_patterns.extend(insight.get('patterns_detected', []))
            all_recommendations.extend(insight.get('strategic_recommendations', []))
            all_meta_obs.extend(insight.get('meta_observations', []))
            
        # Count pattern frequencies
        from collections import Counter
        pattern_counts = Counter(all_patterns)
        recommendation_counts = Counter(all_recommendations)
        
        return {
            'total_insights': len(self.insights_generated),
            'most_common_patterns': pattern_counts.most_common(5),
            'most_common_recommendations': recommendation_counts.most_common(5),
            'total_meta_observations': len(all_meta_obs),
            'avg_confidence': np.mean([ins['confidence'] for ins in self.insights_generated])
        }
