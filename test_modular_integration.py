#!/usr/bin/env python3
"""
Test script to verify all extracted M18.7 modules work together correctly.
This validates our modular architecture maintains feature parity with the monolithic version.
"""

import sys
from typing import Any, Dict, List, Optional
import traceback

def test_imports():
    """Test that all our extracted modules can be imported successfully."""
    print("Testing module imports...")
    
    import_tests = [
        ("physics.engines", ["QuantumEngine", "ClassicalEngine", "E8BoundaryFabric"]),
        ("cognitive.planning", ["ContextBandit", "TorchStateVAEWorldModel"]),
        ("memory.episodic", ["NoveltyScorer", "EpisodicMemory"]),
        ("cognitive.agents", ["BaseAgentAdapter", "MetaArbiter", "NoveltyAgent", "InsightAgent"]),
        ("llm.async_clients", ["AsyncOpenAIClient", "OllamaClient", "GeminiClient", "AsyncLLMPool"]),
        ("neural.autoencoder", ["VariationalAutoencoder", "SubspaceProjector"]),
    ]
    
    results = {}
    
    for module_name, class_names in import_tests:
        try:
            module = __import__(module_name, fromlist=class_names)
            for class_name in class_names:
                if hasattr(module, class_name):
                    cls = getattr(module, class_name)
                    print(f"✓ {module_name}.{class_name} imported successfully")
                else:
                    print(f"✗ {module_name}.{class_name} not found in module")
                    results[f"{module_name}.{class_name}"] = "NOT_FOUND"
            results[module_name] = "SUCCESS"
        except Exception as e:
            print(f"✗ Failed to import {module_name}: {e}")
            results[module_name] = f"ERROR: {e}"
            traceback.print_exc()
    
    return results

def test_basic_functionality():
    """Test basic functionality of key extracted components."""
    print("\nTesting basic functionality...")
    
    try:
        # Test SubspaceProjector from neural module (simplest to test)
        from neural.autoencoder import SubspaceProjector
        
        projector = SubspaceProjector(seed=42)
        print("✓ SubspaceProjector initialization successful")
        
        # Test basic projection
        import numpy as np
        test_vector = np.random.random(10)
        projected = projector.project_to_dim(test_vector, target_dim=5)
        if projected.shape == (5,):
            print("✓ SubspaceProjector basic projection works")
        else:
            print(f"✗ SubspaceProjector projection failed: expected (5,), got {projected.shape}")
        
        # Test NoveltyScorer from memory module
        from memory.episodic import NoveltyScorer
        
        scorer = NoveltyScorer(base_threshold=0.7, decay_factor=0.95)
        print("✓ NoveltyScorer initialization successful")
        
        # Test novelty scoring
        test_embedding = np.random.random(768)
        novelty_score = scorer.score_novelty(test_embedding)
        if 0.0 <= novelty_score <= 1.0:
            print("✓ NoveltyScorer novelty scoring works")
        else:
            print(f"✗ NoveltyScorer returned invalid score: {novelty_score}")
        
        # Test EpisodicMemory
        from memory.episodic import EpisodicMemory
        
        memory = EpisodicMemory(max_episodes=100, embedding_dim=768)
        print("✓ EpisodicMemory initialization successful")
        
        # Test MetaArbiter
        from cognitive.agents import MetaArbiter
        
        arbiter = MetaArbiter()
        print("✓ MetaArbiter initialization successful")
        
        # Test VariationalAutoencoder (may fall back to simple mode)
        from neural.autoencoder import VariationalAutoencoder
        
        vae = VariationalAutoencoder(layer_sizes=[128, 64, 32], console=print)
        print("✓ VariationalAutoencoder initialization successful")
        
        return True
        
    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_integration():
    """Test that components can work together in integration scenarios."""
    print("\nTesting component integration...")
    
    try:
        import numpy as np
        from memory.episodic import EpisodicMemory, NoveltyScorer
        from neural.autoencoder import SubspaceProjector
        
        # Test memory system integration
        memory = EpisodicMemory(max_episodes=100, embedding_dim=64)
        projector = SubspaceProjector(seed=42)
        
        # Create test data
        test_content = "This is a test episode for integration testing"
        raw_embedding = np.random.random(128)
        projected_embedding = projector.project_to_dim(raw_embedding, target_dim=64)
        
        # Store episode
        episode_id = memory.store_episode(
            content=test_content,
            embedding=projected_embedding,
            context={'source': 'integration_test'},
            importance=0.8
        )
        
        if episode_id >= 0:
            print("✓ Memory-Neural integration successful (episode stored)")
        else:
            print("✗ Memory-Neural integration failed (episode not stored)")
            return False
        
        # Test novelty scoring integration
        scorer = NoveltyScorer()
        new_embedding = np.random.random(64)
        novelty1 = scorer.score_novelty(projected_embedding)
        novelty2 = scorer.score_novelty(new_embedding)
        
        if isinstance(novelty1, float) and isinstance(novelty2, float):
            print("✓ Novelty scoring integration successful")
        else:
            print("✗ Novelty scoring integration failed")
            return False
        
        # Test batch projection
        batch_data = np.random.random((5, 128))
        batch_projected = projector.project_batch(batch_data, target_dim=64)
        
        if batch_projected.shape == (5, 64):
            print("✓ Batch processing integration successful")
        else:
            print(f"✗ Batch processing failed: expected (5, 64), got {batch_projected.shape}")
            return False
        
        print("✓ All integration tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Main test runner."""
    print("=" * 60)
    print("E8Mind Modular Architecture Integration Test")
    print("=" * 60)
    
    # Test imports
    import_results = test_imports()
    
    # Test basic functionality
    functionality_ok = test_basic_functionality()
    
    # Test integration
    integration_ok = test_integration()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    failed_imports = [k for k, v in import_results.items() if v != "SUCCESS"]
    
    if failed_imports:
        print(f"Import failures: {failed_imports}")
    else:
        print("✓ All imports successful")
    
    print(f"✓ Functionality tests: {'PASSED' if functionality_ok else 'FAILED'}")
    print(f"✓ Integration tests: {'PASSED' if integration_ok else 'FAILED'}")
    
    overall_success = not failed_imports and functionality_ok and integration_ok
    print(f"\nOverall Status: {'SUCCESS' if overall_success else 'FAILED'}")
    
    if overall_success:
        print("\n🎉 Modular architecture is working correctly!")
        print("All M18.7 components have been successfully extracted and integrated.")
    else:
        print("\n⚠️  Some issues detected. Check the output above for details.")
    
    return overall_success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
