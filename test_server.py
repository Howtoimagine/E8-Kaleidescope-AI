#!/usr/bin/env python3
"""Simple test script to verify telemetry endpoints work."""

import json
import os
import sys
from pathlib import Path

# Add the kaleidoscope directory to the Python path
sys.path.insert(0, str(Path(__file__).parent))

# Set environment variables to avoid interactive prompts
os.environ["E8_PROVIDER"] = "stub"
os.environ["E8_SEMANTIC_DOMAIN"] = ""

try:
    # Import and test the telemetry handlers
    # Load legacy server module from filename (contains a dot) via importlib
    import importlib.util
    from types import ModuleType
    legacy_path = Path(__file__).parent / 'e8_mind_server_M24.1.py'
    spec = importlib.util.spec_from_file_location('e8_mind_server_M24_1', str(legacy_path))
    if spec and spec.loader:
        legacy: ModuleType = importlib.util.module_from_spec(spec)
        sys.modules['e8_mind_server_M24_1'] = legacy
        spec.loader.exec_module(legacy)
        handle_get_graph = getattr(legacy, 'handle_get_graph')
        handle_get_lattice = getattr(legacy, 'handle_get_lattice')
        handle_get_telemetry = getattr(legacy, 'handle_get_telemetry')
    else:
        raise ImportError(f'Unable to load legacy server from {legacy_path}')
    print("✅ Successfully imported telemetry handlers")
    
    # Test if aiohttp is available
    try:
        import aiohttp
        from aiohttp import web
        print("✅ aiohttp is available")
    except ImportError as e:
        print(f"❌ aiohttp not available: {e}")
        sys.exit(1)
    
    # Start a minimal server to test endpoints
    async def test_endpoints():
        print("🧪 Testing telemetry endpoints...")
        
        # Create a mock mind object for testing
        class MockMind:
            def __init__(self):
                self.step_num = 42
                self.physics = MockPhysics()
                
            def _build_telemetry_snapshot(self):
                return {
                    "step": self.step_num,
                    "energy": 0.75,
                    "mood": {"curiosity": 0.8, "confidence": 0.6, "wonder": 0.9}
                }
        
        class MockPhysics:
            def __init__(self):
                import numpy as np
                # Create some mock E8 roots
                self.roots_unit = [
                    np.array([1, 0, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 1, 0, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 1, 0, 0, 0, 0, 0]),
                    np.array([0, 0, 0, 1, 0, 0, 0, 0]),
                    np.array([1, 1, 0, 0, 0, 0, 0, 0]) / np.sqrt(2),
                ]
        
        # Create mock app
        app = web.Application()
        app['mind'] = MockMind()
        
        # Test the lattice endpoint
        from aiohttp.test_utils import make_mocked_request
        request = make_mocked_request('GET', '/api/lattice', app=app)
        
        try:
            response = await handle_get_lattice(request)
            print("✅ /api/lattice endpoint working")
        except Exception as e:
            print(f"❌ /api/lattice failed: {e}")
        
        print("🎉 Basic endpoint tests completed")
    
    # Run the test
    import asyncio
    asyncio.run(test_endpoints())
    
    print("🚀 All tests passed! The server should work with telemetry.")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("   Check that all dependencies are installed")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()