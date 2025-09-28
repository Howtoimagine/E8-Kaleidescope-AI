"""HTTP and WebSocket handlers for E8 Mind (modularized).

This package provides aiohttp request handlers that were previously defined
in the legacy monolithic server file. These handlers expect the aiohttp app
state to include:
- app['mind']: the E8Mind instance
- app['sse_clients']: a set of asyncio.Queue objects used for SSE broadcasting
- app['ws_clients']: a set of aiohttp.web.WebSocketResponse for websocket clients

All handlers are written to be defensive and maintain response shape
compatibility with the legacy endpoints.
"""

from .handlers import (
    handle_get_graph,
    handle_get_graph_summary,
    handle_get_node,
    handle_index,
    handle_memory_search,
    handle_get_state,
    handle_get_telemetry,
    handle_get_blueprint,
    handle_get_lattice,
    handle_stream_telemetry,
    handle_ws_telemetry,
    handle_get_qeng_telemetry,
    handle_get_qeng_ablation,
    handle_get_qeng_probabilities,
    handle_get_metrics_live,
    handle_get_metrics_summary,
    handle_get_metrics_recent,
    handle_post_quantizer,
    handle_post_snapshot,
    handle_add_concept,
    handle_add_concept_legacy,
    handle_trigger_dream,
)  # re-export for ease of import

__all__ = [name for name in globals() if name.startswith('handle_')]