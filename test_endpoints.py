#!/usr/bin/env python3
"""
Test script to verify mock telemetry endpoints
"""
import asyncio
import aiohttp
import json
import os
import argparse

async def test_endpoints(port: int | None = None):
    """Test all HTTP endpoints"""
    if port is None:
        port = int(os.getenv("E8_PORT", "7871"))
    base_url = f"http://127.0.0.1:{port}"
    
    async with aiohttp.ClientSession() as session:
        # Allow server a moment to come up if starting concurrently
        await asyncio.sleep(0.5)

        async def get_with_retry(path: str, retries: int = 20, delay: float = 0.5):
            last_err = None
            for _ in range(retries):
                try:
                    # Return the response without context manager; caller will release
                    response = await session.get(f"{base_url}{path}")
                    return response
                except Exception as e:
                    last_err = e
                    await asyncio.sleep(delay)
            if last_err:
                raise last_err
            raise RuntimeError("GET failed with unknown error")

        # Test /api/lattice
        print("Testing /api/lattice...")
        try:
            resp = await get_with_retry("/api/lattice")
            if resp.status == 200:
                data = await resp.json()
                print(
                    f"OK  /api/lattice: roots={len(data.get('roots_3d', []))}, "
                    f"highlights={len(data.get('active_highlights', []))}"
                )
            else:
                print(f"FAIL /api/lattice: status {resp.status}")
        except Exception as e:
            print(f"ERR  /api/lattice: {e}")
        finally:
            try:
                resp.release()
            except Exception:
                pass

        # Test /api/graph
        print("Testing /api/graph...")
        try:
            resp = await get_with_retry("/api/graph")
            if resp.status == 200:
                data = await resp.json()
                print(
                    f"OK  /api/graph: nodes={len(data.get('nodes', []))}, "
                    f"links={len(data.get('links', []))}"
                )
            else:
                print(f"FAIL /api/graph: status {resp.status}")
        except Exception as e:
            print(f"ERR  /api/graph: {e}")
        finally:
            try:
                resp.release()
            except Exception:
                pass

        # Test /api/telemetry
        print("Testing /api/telemetry...")
        try:
            resp = await get_with_retry("/api/telemetry")
            if resp.status == 200:
                data = await resp.json()
                fs = data.get('field_strength', 0.0)
                coh = data.get('coherence', 0.0)
                try:
                    print(f"OK  /api/telemetry: field_strength={float(fs):.2f}, coherence={float(coh):.2f}")
                except Exception:
                    print(f"OK  /api/telemetry: field_strength={fs}, coherence={coh}")
            else:
                print(f"FAIL /api/telemetry: status {resp.status}")
        except Exception as e:
            print(f"ERR  /api/telemetry: {e}")
        finally:
            try:
                resp.release()
            except Exception:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test E8 Mind server endpoints")
    parser.add_argument("--port", type=int, help="Port of the running server (defaults to E8_PORT or 7871)")
    args = parser.parse_args()
    asyncio.run(test_endpoints(port=args.port))