#!/usr/bin/env python3
"""
Mock telemetry server for testing the frontend integration
Provides the same API endpoints as the E8 Mind server
"""

import asyncio
import json
import time
import random
import math
from aiohttp import web, WSMsgType
from aiohttp_cors import setup as cors_setup, ResourceOptions
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.float32):
            return float(obj)
        return super().default(obj)

def generate_mock_lattice_data():
    """Generate mock E8 lattice data"""
    # Generate 240 E8 root vectors (mock data)
    roots_3d = []
    for i in range(240):
        # Create some pattern in the 3D projection
        angle = i * 2 * math.pi / 240
        radius = 1 + 0.3 * math.sin(i * 0.1)
        x = radius * math.cos(angle) + 0.2 * random.random()
        y = radius * math.sin(angle) + 0.2 * random.random()
        z = 0.5 * math.sin(i * 0.05) + 0.1 * random.random()
        roots_3d.append([x, y, z])
    
    # Mock active highlights (some random subset)
    active_highlights = random.sample(range(240), k=random.randint(10, 30))
    
    # Mock tetrahedron indices
    tetrahedron_indices = [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]]
    
    return {
        "roots_3d": roots_3d,
        "active_highlights": active_highlights,
        "tetrahedron_indices": tetrahedron_indices,
        "step": random.randint(1, 1000),
        "timestamp": time.time()
    }

def generate_mock_graph_data():
    """Generate mock graph data"""
    # Create some nodes
    nodes = []
    for i in range(20):
        nodes.append({
            "id": f"node_{i}",
            "label": f"Concept {i}",
            "x": random.uniform(-100, 100),
            "y": random.uniform(-100, 100),
            "activation": random.uniform(0, 1)
        })
    
    # Create some links
    links = []
    for i in range(30):
        source = random.randint(0, 19)
        target = random.randint(0, 19)
        if source != target:
            links.append({
                "source": f"node_{source}",
                "target": f"node_{target}",
                "weight": random.uniform(0.1, 1.0)
            })
    
    return {
        "nodes": nodes,
        "links": links,
        "timestamp": time.time()
    }

async def handle_get_lattice(request):
    """Mock /api/lattice endpoint"""
    data = generate_mock_lattice_data()
    return web.json_response(data, dumps=lambda obj: json.dumps(obj, cls=NumpyEncoder))

async def handle_get_graph(request):
    """Mock /api/graph endpoint"""
    data = generate_mock_graph_data()
    return web.json_response(data, dumps=lambda obj: json.dumps(obj, cls=NumpyEncoder))

async def handle_get_telemetry(request):
    """Mock /api/telemetry endpoint"""
    data = {
        "lattice": generate_mock_lattice_data(),
        "graph": generate_mock_graph_data(),
        "field_strength": random.uniform(0.5, 2.0),
        "coherence": random.uniform(0.3, 0.9),
        "timestamp": time.time()
    }
    return web.json_response(data, dumps=lambda obj: json.dumps(obj, cls=NumpyEncoder))

async def websocket_handler(request):
    """Mock WebSocket telemetry endpoint"""
    ws = web.WebSocketResponse()
    await ws.prepare(request)
    
    print(f"WebSocket connection established from {request.remote}")
    
    try:
        # Send periodic updates
        while not ws.closed:
            # Send telemetry update every 2 seconds
            telemetry_data = {
                "type": "telemetry_update",
                "lattice": generate_mock_lattice_data(),
                "graph": generate_mock_graph_data(),
                "field_strength": random.uniform(0.5, 2.0),
                "coherence": random.uniform(0.3, 0.9),
                "timestamp": time.time()
            }
            
            await ws.send_str(json.dumps(telemetry_data, cls=NumpyEncoder))
            await asyncio.sleep(2)
            
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("WebSocket connection closed")
    
    return ws

async def create_app():
    """Create the mock server application"""
    app = web.Application()
    
    # Add CORS
    cors = cors_setup(app, defaults={
        "*": ResourceOptions(
            allow_credentials=True,
            expose_headers="*",
            allow_headers="*",
            allow_methods="*"
        )
    })
    
    # Add routes
    app.router.add_get("/api/lattice", handle_get_lattice)
    app.router.add_get("/api/graph", handle_get_graph)  
    app.router.add_get("/api/telemetry", handle_get_telemetry)
    app.router.add_get("/ws/telemetry", websocket_handler)
    
    # Add CORS to all routes
    for route in list(app.router.routes()):
        cors.add(route)
    
    return app

async def main():
    app = await create_app()
    runner = web.AppRunner(app)
    await runner.setup()
    
    site = web.TCPSite(runner, '127.0.0.1', 3001)
    await site.start()
    
    print("🚀 Mock telemetry server started on http://127.0.0.1:3001")
    print("📡 WebSocket endpoint: ws://127.0.0.1:3001/ws/telemetry")
    print("📊 HTTP endpoints:")
    print("   GET /api/lattice - E8 root lattice data")  
    print("   GET /api/graph - Concept graph data")
    print("   GET /api/telemetry - Combined telemetry data")
    print("\n🔄 WebSocket sends updates every 2 seconds")
    print("Press Ctrl+C to stop...")
    
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        print("\n👋 Shutting down mock server...")
        await runner.cleanup()

if __name__ == "__main__":
    asyncio.run(main())