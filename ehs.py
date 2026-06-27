import asyncio
import threading
import queue
import time
from dataclasses import dataclass
from typing import Any, Dict, Callable, Tuple


@dataclass
class GeoOp:
    op: str
    args: Dict[str, Any]
    fut: asyncio.Future
    deps: Tuple[int, ...] = ()


class EventHorizonScheduler:
    """Lightweight threaded geometric event queue that bridges to asyncio.

    Designed as a drop-in from the design notes. Workers run in threads and
    execute geometry-heavy blocking operations. Async callers use
    `await ehs.submit(op, **kwargs)` to schedule work and receive the result.
    """
    def __init__(self, mind, n_workers: int = 2, max_queue: int = 1024):
        self.mind = mind
        self.q: "queue.Queue[GeoOp]" = queue.Queue(maxsize=max_queue)
        # thread-local marker used to indicate the worker thread is the authorized
        # geometry writer. Other code can check this to enforce single-writer invariants
        import threading as _threading
        self._thread_local = _threading.local()
        self._workers = []
        self._op_id = 0
        self._lock = threading.Lock()
        self._handlers: Dict[str, Callable[[Dict[str, Any]], Any]] = {
            "ADD_NODE": self._h_add_node,
            "SHELL_SPIN": self._h_shell_spin,
            "UPDATE_SHELL_INDEX": self._h_update_shell_index,
            "ANNEAL_MAIN_INDEX": self._h_anneal_main_index,
            "GRAPH_MOD": self._h_graph_mod,
        }
        self._n_workers = max(1, int(n_workers or 1))

    def start(self):
        for i in range(self._n_workers):
            t = threading.Thread(target=self._worker, daemon=True)
            t.start()
            self._workers.append(t)

    async def submit(self, op: str, **args):
        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        # Put will block if queue full; do not block event loop
        try:
            self.q.put_nowait(GeoOp(op=op, args=args, fut=fut))
        except queue.Full:
            # Backpressure: await briefly and retry once
            await asyncio.sleep(0.01)
            try:
                self.q.put_nowait(GeoOp(op=op, args=args, fut=fut))
            except queue.Full:
                raise RuntimeError("EventHorizonScheduler queue full")
        return await fut

    # ---------- worker & handlers ----------
    def _worker(self):
        while True:
            op = self.q.get()
            # mark this worker thread as the authorized geometry writer
            try:
                self._thread_local.is_geo_writer = True
            except Exception:
                pass
            try:
                fn = self._handlers.get(op.op)
                res = fn(op.args) if fn else None
                self._resolve(op.fut, res)
            except Exception as e:
                # ensure exceptions propagate back to the awaiting coroutine
                try:
                    self._reject(op.fut, e)
                except Exception:
                    pass
            finally:
                # clear the writer mark to avoid accidental propagation
                try:
                    self._thread_local.is_geo_writer = False
                except Exception:
                    pass

    def _resolve(self, fut, val):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            # set result from non-async thread
            asyncio.run(self._set_from_thread(fut, val))
            return
        loop.call_soon_threadsafe(lambda: fut.set_result(val))

    def _reject(self, fut, err):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self._set_exc_from_thread(fut, err))
            return
        loop.call_soon_threadsafe(lambda: fut.set_exception(err))

    async def _set_from_thread(self, fut, val):
        fut.set_result(val)

    async def _set_exc_from_thread(self, fut, err):
        fut.set_exception(err)

    # --- geometry operations (thread realm) ---
    def _h_add_node(self, a):
        return self.mind.memory._add_node_geometric_locked(**a)

    def _h_shell_spin(self, a):
        dim = int(a.get("dim"))
        bcoef = a.get("bcoef")
        angle = float(a.get("angle", 0.0))
        shell = self.mind.dimensional_shells.get(dim)
        if shell and hasattr(shell, "spin_with_bivector"):
            shell.spin_with_bivector(bcoef, angle)
        return True

    def _h_update_shell_index(self, a):
        dim = int(a.get("dim"))
        shell = self.mind.dimensional_shells.get(dim)
        if shell and hasattr(self.mind, "proximity_engine"):
            try:
                self.mind.proximity_engine.update_shell_index(shell.dim, shell)
            except Exception:
                pass
        return True

    def _h_anneal_main_index(self, a):
        # Commit pending additions under memory's existing lock/contract
        # Commit pending additions under memory's existing lock/contract
        self.mind.memory._commit_pending_additions_locked()
        # Also perform lightweight HoloIndex leaf-only anneal if present
        try:
            if hasattr(self.mind.memory, 'anneal_holo_leafs'):
                self.mind.memory.anneal_holo_leafs()
        except Exception:
            pass
        return True

    def _h_graph_mod(self, a):
        return self.mind.memory.graph_db.add_edge(**a)
