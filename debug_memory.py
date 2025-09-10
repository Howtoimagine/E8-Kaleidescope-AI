#!/usr/bin/env python3
"""
Debug script to diagnose memory and concept issues in E8Mind.
"""

import asyncio
import sys
import json
import importlib.util

# Import from the main server file
def load_e8mind_module():
    spec = importlib.util.spec_from_file_location("e8_mind_server", "e8_mind_server_M18.7.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

e8_module = load_e8mind_module()

async def debug_memory():
    """Run diagnostic checks on the E8Mind memory system."""
    print("🔍 E8Mind Memory Diagnostic Tool")
    print("=" * 50)
    
    try:
        # Initialize E8Mind instance
        print("Initializing E8Mind...")
        mind = E8Mind()
        
        # Get detailed memory statistics
        print("\n📊 Memory Statistics:")
        stats = mind.get_detailed_memory_stats()
        
        for key, value in stats.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for subkey, subvalue in value.items():
                    print(f"    {subkey}: {subvalue}")
            else:
                print(f"  {key}: {value}")
        
        # Check ArXiv data sources
        print("\n📡 Data Sources Configuration:")
        from e8_mind_server_M18_7 import DATA_SOURCES
        for name, config in DATA_SOURCES.items():
            print(f"  {name}: {config}")
        
        # Check ingestion pipeline state
        if hasattr(mind, 'ingestion_pipeline'):
            print(f"\n🔄 Ingestion Pipeline:")
            print(f"  Sources: {len(mind.ingestion_pipeline.sources)}")
            print(f"  Running: {getattr(mind.ingestion_pipeline, 'running', False)}")
        
        # Test concept addition
        print("\n🧪 Testing Concept Addition...")
        test_entry = {
            "type": "test_concept",
            "label": "Debug Test Concept",
            "metaphor": "This is a test concept for debugging the memory system.",
            "rating": 0.8,
            "step": 0,
            "source": "debug_script"
        }
        
        concept_id = await mind.memory.add_entry(test_entry)
        print(f"  Added test concept with ID: {concept_id}")
        
        # Recheck stats after addition
        new_stats = mind.get_detailed_memory_stats()
        print(f"  Graph nodes after addition: {new_stats['graph_nodes']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during diagnosis: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = asyncio.run(debug_memory())
    sys.exit(0 if success else 1)
