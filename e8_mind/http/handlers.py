from __future__ import annotations

import asyncio
import contextlib
import json
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from aiohttp import web, WSMsgType  # type: ignore
except Exception:  # pragma: no cover
    from typing import Any as _Any
    web: _Any = None  # type: ignore
    WSMsgType: _Any = None  # type: ignore

# Prefer modular utils for encoders and file paths. Use local aliases to avoid type conflicts
try:
    from e8_mind.core.utils import NumpyEncoder as JSONEncoderAdapter, get_path as util_get_path, sanitize_line as util_sanitize_line  # type: ignore
except Exception:  # fallback to legacy-compatible shims
    class JSONEncoderAdapter(json.JSONEncoder):
        def default(self, obj):
            try:
                import numpy as _np
                if isinstance(obj, _np.ndarray):
                    return obj.tolist()
                if isinstance(obj, _np.integer):
                    return int(obj)
                if isinstance(obj, _np.floating):
                    return float(obj)
            except Exception:
                pass
            return super().default(obj)
    def util_get_path(rel: str, run_id: str, runtime_dir: str) -> str:  # fallback signature matches modular util
        base = os.path.join(runtime_dir or 'runtime', str(run_id)) if run_id else (runtime_dir or 'runtime')
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, rel)
    def util_sanitize_line(text: str, max_chars: int = 80) -> str:
        if not isinstance(text, str):
            return ""
        return text.replace('\n',' ').replace('\r','').strip()[:max_chars]

try:
    from e8_mind.physics.e8 import triacontagonal_projection  # type: ignore
except Exception:  # pragma: no cover
    def triacontagonal_projection(_roots, _console=None):
        return np.asarray([])

try:
    from e8_mind.core.orchestrator import GLOBAL_SEED as ORCHESTRATOR_GLOBAL_SEED
    from e8_mind.core.orchestrator import deterministic_embedding_stub as modular_deterministic_embedding
except Exception:  # pragma: no cover - orchestrator unavailable
    ORCHESTRATOR_GLOBAL_SEED = 1337
    modular_deterministic_embedding = None

# Graph export utility (kept local to avoid heavy nx import at module top)
def _export_graph(graph: Any) -> Dict[str, Any]:
    try:
        import networkx as nx  # noqa: F401
        from networkx.readwrite import json_graph
    except Exception:
        return {"nodes": [], "links": []}
    try:
        # Preserve current JSON shape with 'links' while avoiding the FutureWarning
        return json_graph.node_link_data(graph, edges="links")
    except TypeError:
        return json_graph.node_link_data(graph)

async def handle_get_graph(request):
    mind = request.app['mind']
    data = _export_graph(mind.memory.graph_db.graph)
    try:
        for n in data.get('nodes') or []:
            if not n.get('label'):
                n['label'] = n.get('name') or n.get('title') or n.get('id') or '(unnamed concept)'
    except Exception:
        pass
    return web.Response(text=json.dumps(data, cls=JSONEncoderAdapter), content_type='application/json')

async def handle_get_node(request):
    try:
        node_id = request.match_info['node_id']
        mind = request.app['mind']
        G = mind.memory.graph_db.graph
        if not G.has_node(node_id):
            return web.json_response({"error": "Node not found"}, status=404)
        node_data = dict(G.nodes[node_id]); node_data['id'] = node_id
        neighbors, edges = [], []
        for nb in G.neighbors(node_id):
            d = dict(G.nodes[nb]); d['id'] = nb; neighbors.append(d)
        for s, t, d in G.edges(node_id, data=True):
            edges.append({'source': s, 'target': t, 'data': d})
        return web.json_response({
            'node': node_data,
            'neighbors': neighbors[:8],
            'edges': edges,
            'total_neighbors': len(neighbors)
        }, dumps=lambda d: json.dumps(d, cls=JSONEncoderAdapter))
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def handle_ws_telemetry(request):
    mind = request.app['mind']
    client_ip = request.remote or 'unknown'
    try:
        ws = web.WebSocketResponse(heartbeat=30.0, timeout=60.0)
    except TypeError:
        try:
            ws = web.WebSocketResponse(heartbeat=30.0)
        except TypeError:
            ws = web.WebSocketResponse()
    try:
        await ws.prepare(request)
    except Exception:
        return ws
    try:
        request.app['ws_clients'].add(ws)
        mind.ws_clients = request.app['ws_clients']
        snap = mind._build_telemetry_snapshot()
        await ws.send_str(json.dumps(snap, cls=JSONEncoderAdapter, ensure_ascii=False))
    except Exception:
        pass
    try:
        async for msg in ws:
            if WSMsgType and msg.type == WSMsgType.TEXT:
                data = (msg.data or '').strip().lower()
                if data in {"close", "quit", "bye"}:
                    await ws.close(); break
                if data == "ping":
                    with contextlib.suppress(Exception):
                        await ws.send_str("pong")
            elif WSMsgType and msg.type in (WSMsgType.ERROR, WSMsgType.CLOSE):
                break
    except asyncio.CancelledError:
        pass
    except Exception:
        pass
    finally:
        with contextlib.suppress(Exception):
            request.app['ws_clients'].discard(ws)
            if not ws.closed:
                await ws.close()
    return ws

async def handle_get_graph_summary(request):
    mind = request.app['mind']
    G = mind.memory.graph_db.graph
    try:
        node_count, edge_count = G.number_of_nodes(), G.number_of_edges()
        recent = []
        for nid in list(getattr(mind.memory, 'recent_nodes', []))[-25:][::-1]:
            try:
                data = G.nodes[nid]
            except Exception:
                continue
            lab = data.get('label')
            if lab:
                recent.append({'id': nid, 'label': lab, 'type': data.get('type'), 'rating': data.get('rating'), 'temperature': data.get('temperature')})
            if len(recent) >= 10:
                break
        return web.json_response({'nodes': node_count, 'edges': edge_count, 'recent': recent})
    except Exception as e:
        return web.json_response({'error': str(e)}, status=500)

INDEX_HTML = r"""<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'/><title>E8 Mind Console</title></head><body><h3>E8 Mind Server</h3><p>Static UI not found. API is running.</p></body></html>"""

async def handle_index(request):
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    static_idx = os.path.join(base_dir, 'static', 'index.html')
    if os.path.exists(static_idx):
        return web.FileResponse(static_idx)
    return web.Response(text=INDEX_HTML, content_type='text/html')

async def handle_get_qeng_telemetry(request):
    mind = request.app['mind']
    qeng = getattr(mind, 'qeng', None)
    if qeng is None:
        return web.json_response({"error": "quantum engine not initialized"}, status=400)
    return web.json_response(qeng.telemetry_state())

async def handle_stream_telemetry(request):
    app = request.app
    q: asyncio.Queue[str | None] = asyncio.Queue(maxsize=16)
    app['sse_clients'].add(q)
    headers = {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'X-Accel-Buffering': 'no',
    }
    resp = web.StreamResponse(status=200, reason='OK', headers=headers)
    await resp.prepare(request)
    try:
        await resp.write(b":ok\n\n")
        try:
            mind = app.get('mind')
            if mind is not None:
                init_payload = json.dumps(mind._build_telemetry_snapshot(), cls=JSONEncoderAdapter, ensure_ascii=False)
                with contextlib.suppress(asyncio.QueueFull):
                    q.put_nowait(init_payload)
        except Exception:
            pass
        async def _heartbeat_writer():
            try:
                while True:
                    await asyncio.sleep(25)
                    try:
                        hb_chunk = f"event: heartbeat\ndata: {{\"ts\": {int(time.time())}}}\n\n".encode('utf-8')
                        await resp.write(hb_chunk)
                    except Exception:
                        break
            except asyncio.CancelledError:
                pass
        hb_task = asyncio.create_task(_heartbeat_writer())
        while True:
            data = await q.get()
            if data is None:
                break
            try:
                evt, data_txt = "telemetry", data
                try:
                    obj = json.loads(data)
                    if isinstance(obj, dict):
                        t = obj.get("type")
                        if t == "tetra_update":
                            evt, data_txt = "tetra_update", json.dumps(obj, ensure_ascii=False)
                        elif t == "curvature":
                            evt, data_txt = "curvature", json.dumps(obj, ensure_ascii=False)
                except Exception:
                    pass
                chunk = f"event: {evt}\ndata: {data_txt}\n\n".encode('utf-8')
            except Exception:
                chunk = f"event: telemetry\ndata: {data}\n\n".encode('utf-8')
            try:
                await resp.write(chunk)
            except Exception:
                break
        hb_task.cancel()
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        pass
    finally:
        app['sse_clients'].discard(q)
        with contextlib.suppress(Exception):
            await resp.write_eof()
    return resp

async def handle_get_qeng_ablation(request):
    mind = request.app['mind']
    qeng = getattr(mind, 'qeng', None)
    if qeng is None:
        return web.json_response({"error": "quantum engine not initialized"}, status=400)
    params = request.rel_url.query
    prev_idx = int(params.get('prev_idx', '0'))
    sigma = float(params.get('sigma')) if params.get('sigma') is not None else None
    window = int(params.get('window', '5'))
    trials = int(params.get('trials', '256'))
    res = qeng.measure_ablation(prev_idx=prev_idx, sigma=sigma, window=window, trials=trials)
    return web.json_response(res)

async def handle_get_qeng_probabilities(request):
    mind = request.app['mind']
    qeng = getattr(mind, 'qeng', None)
    if qeng is None:
        return web.json_response({"error": "quantum engine not initialized"}, status=400)
    try:
        topk = int(request.rel_url.query.get("topk", "0"))
    except ValueError:
        topk = 0
    try:
        probs = None
        if hasattr(qeng, "_probs"):
            probs = qeng._probs()
        elif hasattr(qeng, "probs"):
            probs = qeng.probs()
        if probs is None:
            return web.json_response({"error": "probabilities unavailable"}, status=500)
        p = np.asarray(probs, dtype=np.float32).ravel()
        if p.size == 0:
            return web.json_response({"error": "empty probability vector"}, status=500)
        s = float(p.sum())
        if s <= 0:
            return web.json_response({"error": "invalid probability sum"}, status=500)
        p = (p / s).astype(np.float32)
        if topk > 0:
            k = min(topk, p.size)
            idxs = np.argsort(-p)[:k]
            data = [{"index": int(i), "prob": float(p[i])} for i in idxs]
            return web.json_response({"topk": k, "distribution": data})
        else:
            return web.json_response({"distribution": [float(x) for x in p.tolist()]})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def handle_memory_search(request):
    mind = request.app['mind']
    q_text = request.rel_url.query.get('q', '').strip()
    try:
        k = int(request.rel_url.query.get('k', '5'))
    except ValueError:
        k = 5
    if not q_text:
        return web.json_response({"error": "missing query param 'q'"}, status=400)
    try:
        vec = await mind.get_embedding(q_text)
    except Exception:
        embed_dim = getattr(mind, 'embed_in_dim', 1536)
        if modular_deterministic_embedding is not None:
            try:
                raw = modular_deterministic_embedding(q_text, embed_dim, ORCHESTRATOR_GLOBAL_SEED)
                vec = mind.embed_adapter(raw)
            except Exception:
                vec = np.zeros(embed_dim, dtype=np.float32)
        else:
            vec = np.zeros(embed_dim, dtype=np.float32)
    try:
        sims = mind.memory.find_similar_in_main_storage_e8(vec, k=k, decode_remnants=True)
    except Exception as e:
        return web.json_response({"error": f"search failed: {e}"}, status=500)
    out = []
    for nid, dist in sims:
        node = mind.memory.graph_db.get_node(nid) or {}
        out.append({
            "id": nid,
            "label": node.get("label"),
            "type": node.get("type"),
            "rating": node.get("rating"),
            "temperature": node.get("temperature"),
            "distance": float(dist)
        })
    return web.json_response({"q": q_text, "k": k, "results": out})

async def handle_get_metrics_live(request):
    mind = request.app['mind']
    metrics_file = util_get_path("metrics.ndjson", mind.run_id, runtime_dir=os.getenv('RUNTIME_DIR', 'runtime') or 'runtime')
    if not os.path.exists(metrics_file):
        return web.json_response({"error": "metrics file not found"}, status=404)
    try:
        tail_lines = 400
        lines: list[str] = []
        with open(metrics_file, "r", encoding="utf-8") as f:
            for line in f:
                lines.append(line)
                if len(lines) > tail_lines:
                    lines.pop(0)
        import collections
        counters = collections.defaultdict(int)
        gauges: dict[str, float] = {}
        timings = collections.defaultdict(list)
        for ln in lines:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            t = rec.get("type")
            if t == "counter":
                counters[rec.get("name","?")] += int(rec.get("value", 0))
            elif t == "gauge":
                gauges[rec.get("name","?")] = float(rec.get("value", 0.0))
            elif t == "timing":
                timings[rec.get("name","?")].append(float(rec.get("duration_ms", 0.0)))
        timing_means = {k: (sum(v)/len(v) if v else 0.0) for k, v in timings.items()}
        return web.json_response({"counters": dict(counters), "gauges": gauges, "timing_means": timing_means})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def handle_get_metrics_summary(request):
    mind = request.app['mind']
    metrics_file = util_get_path("metrics.ndjson", mind.run_id, runtime_dir=os.getenv('RUNTIME_DIR', 'runtime') or 'runtime')
    if not os.path.exists(metrics_file):
        return web.json_response({"error": "metrics file not found"}, status=404)
    import collections
    counters = collections.defaultdict(int)
    gauges: dict[str, float] = {}
    timings = collections.defaultdict(list)
    try:
        with open(metrics_file, "r", encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    rec = json.loads(ln)
                except Exception:
                    continue
                t = rec.get("type")
                if t == "counter":
                    counters[rec.get("name","?")] += int(rec.get("value", 0))
                elif t == "gauge":
                    gauges[rec.get("name","?")] = float(rec.get("value", 0.0))
                elif t == "timing":
                    timings[rec.get("name","?")].append(float(rec.get("duration_ms", 0.0)))
        timing_stats = {}
        for name, vals in timings.items():
            if not vals:
                continue
            timing_stats[name] = {
                "count": len(vals),
                "mean": float(sum(vals)/len(vals)),
                "min": float(min(vals)),
                "max": float(max(vals))
            }
        return web.json_response({"counters": dict(counters), "gauges": gauges, "timings": timing_stats})
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def handle_post_quantizer(request):
    mind = request.app['mind']
    try:
        data = await request.json()
    except Exception:
        return web.json_response({"error": "invalid JSON"}, status=400)
    qtype = (data.get("type") or "").lower()
    allowed = {"e8", "cubic", "random", "none"}
    if qtype not in allowed:
        return web.json_response({"error": f"invalid type; must be one of {sorted(allowed)}"}, status=400)
    mind._quantizer_override = qtype
    try:
        console = getattr(mind, 'console', None)
        if console:
            console.log(f"[Quantizer] override set -> {qtype}")
    except Exception:
        pass
    return web.json_response({"status": "ok", "quantizer": qtype})

async def handle_post_snapshot(request):
    mind = request.app['mind']
    try:
        console = getattr(mind, 'console', None)
        if console:
            console.log("📸 Snapshot requested via API.")
    except Exception:
        pass
    asyncio.create_task(mind.memory.snapshot())
    return web.json_response({"status": "ok", "message": "snapshot initiated"})

async def handle_get_state(request):
    mind = request.app['mind']
    try:
        snap = mind._build_telemetry_snapshot()
    except Exception:
        snap = {}
    state = {
        'step': getattr(mind, 'step_num', None),
        'mood': getattr(getattr(mind, 'mood', None), 'mood_vector', None),
        'insight_reward': getattr(mind, 'last_insight_reward', None),
        'goals': getattr(mind, 'active_goals', []),
        'telemetry': snap
    }
    return web.json_response(state, dumps=lambda d: json.dumps(d, cls=JSONEncoderAdapter))

async def handle_get_telemetry(request):
    mind = request.app['mind']
    try:
        telemetry_data = mind._build_telemetry_snapshot()
        if getattr(mind, 'market', None):
            market_obj = telemetry_data.get("market")
            if not isinstance(market_obj, dict):
                market_obj = {}
                telemetry_data["market"] = market_obj
            market_obj["bars"] = {
                "1s": {s: list(mind.market.history_1s.get(s, [])) for s in mind.market_symbols},
                "1m": {s: list(mind.market.history_1m.get(s, [])) for s in mind.market_symbols},
            }
        return web.json_response(telemetry_data, dumps=lambda d: json.dumps(d, cls=JSONEncoderAdapter))
    except Exception as e:
        try:
            console = getattr(mind, 'console', None)
            if console:
                console.log(f"[Telemetry Endpoint Error] {e}")
        except Exception:
            pass
        # Be resilient for frontend seeding: return an empty object instead of 500
        # Frontend will continue with SSE/WS updates; this avoids noisy XHR 500s on cold start
        return web.json_response({}, status=200, dumps=lambda d: json.dumps(d, cls=JSONEncoderAdapter))

async def handle_get_blueprint(request):
    return web.json_response(request.app['mind'].blueprint)

async def handle_get_lattice(request):
    mind = request.app['mind']
    try:
        lattice_data: Dict[str, Any] = {
            "roots": [],
            "connections": [],
            "thinking_traces": [],
            "active_roots": [],
            # Legacy-compatible aliases (populated before response):
            # - roots_3d: list of [x, y, z]
            # - active_highlights: list of indices into roots_3d/roots
            "roots_3d": [],
            "active_highlights": [],
            "tetrahedron": [],
            "meta": {
                "step": getattr(mind, 'step_num', 0),
                "energy": 0.0,
                "active_dimension": 8,
                "total_roots": 0,
                "shell_population": {},
                "shell_tension": {},
            },
        }

        physics = getattr(mind, 'physics', None)
        projection_matrix: Optional[np.ndarray] = None
        root_vectors: Optional[np.ndarray] = None
        root_positions: Optional[np.ndarray] = None
        max_roots = 240

        if physics is not None and hasattr(physics, 'roots'):
            try:
                root_vectors = np.asarray(getattr(physics, 'roots', []), dtype=np.float32)
            except Exception:
                root_vectors = None
            if root_vectors is not None and root_vectors.size > 0:
                # Prefer cached projection matrix if physics has already generated one
                projection_matrix = getattr(physics, 'projection_matrix', None)
                if projection_matrix is not None:
                    try:
                        projection_matrix = np.asarray(projection_matrix, dtype=np.float32)
                        if projection_matrix.shape[0] != 8 or projection_matrix.shape[1] < 3:
                            projection_matrix = None
                    except Exception:
                        projection_matrix = None
                if projection_matrix is None:
                    try:
                        # fall back to triacontagonal projection used during initialization
                        root_positions = triacontagonal_projection(root_vectors, getattr(mind, 'console', None))
                    except Exception:
                        root_positions = None
                if root_positions is None:
                    try:
                        # Simple orthographic slice if projection failed
                        root_positions = root_vectors[:, :3]
                    except Exception:
                        root_positions = None
                else:
                    # triacontagonal_projection already returns scaled positions
                    pass

                if projection_matrix is not None and root_positions is None:
                    try:
                        root_positions = root_vectors @ projection_matrix[:, :3]
                    except Exception:
                        root_positions = None

                if root_positions is not None and root_positions.size > 0:
                    # normalize spread for visualization
                    span = float(np.max(np.linalg.norm(root_positions, axis=1))) or 1.0
                    scale = 6.0 / span
                    root_positions = (root_positions * scale).astype(np.float32)

                    total_roots = min(root_positions.shape[0], max_roots)
                    weights_matrix = None
                    adj_bool = None
                    try:
                        weights_matrix = np.asarray(getattr(physics, 'weights', None), dtype=np.float32)
                    except Exception:
                        weights_matrix = None
                    try:
                        adj_bool = np.asarray(getattr(physics, 'adj_bool', None), dtype=np.int8)
                    except Exception:
                        adj_bool = None

                    roots_payload: list[Dict[str, Any]] = []
                    for idx in range(total_roots):
                        vec8 = root_vectors[idx].tolist()
                        pos3 = root_positions[idx].tolist()
                        parity = 'even' if int(sum(vec8)) % 2 == 0 else 'odd'
                        degree = 0
                        neighbors: list[int] = []
                        if adj_bool is not None and adj_bool.shape[0] > idx:
                            neigh_idx = np.where(adj_bool[idx][:total_roots] > 0)[0]
                            neighbors = [int(n) for n in np.asarray(neigh_idx, dtype=int)]
                            degree = len(neighbors)
                        energy = float(np.linalg.norm(root_vectors[idx]))
                        roots_payload.append({
                            "id": idx,
                            "position": [float(pos3[0]), float(pos3[1]), float(pos3[2])],
                            "vector8": [float(v) for v in vec8],
                            "shell": float(np.linalg.norm(root_vectors[idx])),
                            "energy": energy,
                            "parity": parity,
                            "degree": degree,
                            "neighbors": neighbors,
                        })
                    lattice_data["roots"] = roots_payload
                    lattice_data["meta"]["total_roots"] = total_roots
                    # Build legacy flat positions
                    lattice_data["roots_3d"] = [r["position"] for r in roots_payload]

                    # Build connection list from weights
                    if weights_matrix is not None and weights_matrix.size > 0:
                        conn: list[Dict[str, Any]] = []
                        capped = min(weights_matrix.shape[0], total_roots)
                        for i in range(capped):
                            row = weights_matrix[i][:capped]
                            for j in range(i + 1, capped):
                                w = float(row[j])
                                if w <= 0:
                                    continue
                                conn.append({
                                    "source": int(i),
                                    "target": int(j),
                                    "weight": w,
                                })
                        # keep strongest connections to limit payload size
                        conn.sort(key=lambda item: item["weight"], reverse=True)
                        lattice_data["connections"] = conn[:2500]

                    # active roots derived from time-dependent sweep
                    step = getattr(mind, 'step_num', 0)
                    active_count = min(16, total_roots)
                    active_roots = []
                    if active_count > 0:
                        for offset in range(active_count):
                            idx = (step + offset * 17) % total_roots
                            active_roots.append(int(idx))
                    lattice_data["active_roots"] = active_roots
                    lattice_data["active_highlights"] = list(active_roots)

                    # Add a simple tetrahedron skeleton anchored to first four roots for reference
                    if total_roots >= 4:
                        lattice_data["tetrahedron"] = [0, 1, 2, 3]

        # Shell metrics (population & tension) derived from dimensional shells
        dimensional_shells = getattr(mind, 'dimensional_shells', {}) or {}
        shell_population: Dict[str, int] = {}
        shell_tension: Dict[str, float] = {}
        for dim, shell in dimensional_shells.items():
            matrix, _ = shell.get_all_vectors_as_matrix()
            dim_key = str(dim)
            if matrix is None or getattr(matrix, 'size', 0) == 0:
                shell_population[dim_key] = len(getattr(shell, 'vectors', {}))
                shell_tension[dim_key] = 0.0
                continue
            matrix = np.asarray(matrix, dtype=np.float32)
            shell_population[dim_key] = int(matrix.shape[0])
            if matrix.shape[0] > 1:
                center = np.mean(matrix, axis=0, keepdims=True)
                dists = np.linalg.norm(matrix - center, axis=1)
                shell_tension[dim_key] = float(np.mean(dists))
            else:
                shell_tension[dim_key] = 0.0
        lattice_data["meta"]["shell_population"] = shell_population
        lattice_data["meta"]["shell_tension"] = shell_tension
        lattice_data["meta"]["energy"] = float(getattr(mind, 'last_insight_reward', 0.0))

        # Sample recent cognitive activity and project onto the lattice
        memory = getattr(mind, 'memory', None)
        thoughts: list[Dict[str, Any]] = []
        if memory is not None:
            main_vectors = getattr(memory, 'main_vectors', {}) or {}
            recent_ids = list(getattr(memory, 'recent_nodes', []))[-24:]
            physics_project = getattr(physics, 'find_nearest_root_index', None)
            for node_id in reversed(recent_ids):
                node_data = {}
                try:
                    node_data = memory.graph_db.get_node(node_id) or {}
                except Exception:
                    node_data = {}
                label = node_data.get('label') or node_data.get('metaphor') or node_id
                rating = node_data.get('rating')
                vec = main_vectors.get(node_id)
                if vec is None:
                    continue
                try:
                    vec8 = memory.project_to_dim8(vec).tolist()
                except Exception:
                    try:
                        vec8 = list(np.asarray(vec, dtype=np.float32)[:8])
                    except Exception:
                        continue
                root_idx = None
                if callable(physics_project):
                    try:
                        root_idx = physics_project(np.asarray(vec8, dtype=np.float32))
                    except Exception:
                        root_idx = None
                if root_idx is None and root_vectors is not None and root_vectors.size > 0:
                    try:
                        diffs = root_vectors[:max_roots] - np.asarray(vec8, dtype=np.float32)
                        norms = np.linalg.norm(diffs, axis=1)
                        root_idx = int(np.argmin(norms))
                    except Exception:
                        root_idx = None
                root_idx = int(root_idx) if root_idx is not None else None
                projection = None
                if root_idx is not None and root_positions is not None and root_positions.shape[0] > root_idx:
                    projection = root_positions[root_idx].tolist()
                thoughts.append({
                    "node_id": node_id,
                    "label": label,
                    "rating": float(rating) if isinstance(rating, (int, float)) else None,
                    "vector8": [float(v) for v in vec8],
                    "root_index": root_idx,
                    "projection": [float(v) for v in projection] if projection is not None else None,
                })
        lattice_data["thinking_traces"] = thoughts[:18]

        return web.json_response(lattice_data, dumps=lambda d: json.dumps(d, cls=JSONEncoderAdapter))
    except Exception as e:
        try:
            console = getattr(mind, 'console', None)
            if console:
                console.log(f"[Lattice API Error] {e}")
        except Exception:
            pass
        return web.json_response({"error": "Failed to generate lattice data"}, status=500)

async def handle_get_bh_panel(request):
    mind = request.app['mind']
    try:
        mass = float(request.query.get('mass', os.getenv("E8_BH_PANEL_MASS", "1.0")))
        dim = int(request.query.get('dim', os.getenv("E8_BH_PANEL_DIM", "32")))
        panel_data = getattr(mind, '_last_bh_panel', None)
        if not panel_data:
            panel_data = mind.bh_panel_snapshot(mass=mass, dim=dim)
        return web.json_response(panel_data)
    except Exception as e:
        try:
            console = getattr(mind, 'console', None)
            if console:
                console.log(f"[BH Panel API Error] {e}")
        except Exception:
            pass
        return web.json_response({"error": "Failed to generate BH panel data"}, status=500)

async def handle_get_metrics_recent(request):
    try:
        runtime_dir = os.getenv('RUNTIME_DIR', 'runtime')
        metrics_file = os.path.join(runtime_dir, "metrics.ndjson")
        recent_metrics = []
        if os.path.exists(metrics_file):
            try:
                max_lines = int(request.query.get('limit', '100'))
                with open(metrics_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    for line in lines[-max_lines:]:
                        try:
                            metric = json.loads(line.strip())
                            recent_metrics.append(metric)
                        except json.JSONDecodeError:
                            continue
            except Exception:
                pass
        return web.json_response({"metrics": recent_metrics, "count": len(recent_metrics)})
    except Exception:
        return web.json_response({"error": "Failed to retrieve recent metrics"}, status=500)

async def handle_add_concept(request):
    mind = request.app['mind']
    try:
        data = await request.json()
        text = data.get("text")
        if not text:
            return web.json_response({"error": "Text is required"}, status=400)
        rating = await mind.rate_concept(text)
        entry = {"type": "external_concept", "label": util_sanitize_line(text, 25), "metaphor": text, "rating": rating, "step": mind.step_num}
        node_id = await mind.memory.add_entry(entry)
        return web.json_response({"node_id": node_id, "message": "Concept added successfully"})
    except json.JSONDecodeError:
        return web.json_response({"error": "Invalid JSON"}, status=400)
    except Exception as e:
        return web.json_response({"error": str(e)}, status=500)

async def handle_trigger_dream(request):
    mind = request.app['mind']
    asyncio.create_task(mind.dream_engine.run_dream_sequence())
    return web.json_response({"status": "Dream sequence initiated"})

# --- Graceful shutdown hooks (modular) ---
async def shutdown_sse(app):
    try:
        clients = app.get('sse_clients') or set()
        for q in list(clients):
            try:
                q.put_nowait(None)
            except Exception:
                pass
    except Exception:
        pass

async def shutdown_market_feed(app):
    try:
        mind = app.get('mind')
        if mind and getattr(mind, 'market', None):
            await mind.market.stop()
    except Exception:
        pass

async def shutdown_ws(app):
    try:
        ws_set = app.get('ws_clients')
        if not ws_set:
            return
        for ws in list(ws_set):
            try:
                await ws.close(code=1001, message='Server shutdown')
            except Exception:
                pass
    except Exception:
        pass
