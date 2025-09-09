from typing import Any, List, Optional, Dict
from dataclasses import dataclass

import numpy as np
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

# ----- Request/Response Models -----

class Anchor(BaseModel):
    vector: List[float] = Field(..., min_length=8, max_length=8)  # 8D
    weight: float = 1.0

class SampleRequest(BaseModel):
    anchors: List[Anchor]
    top_k: int = 5
    curiosity_alpha: float = 0.12
    curiosity_visits: Optional[List[float]] = None
    weyl_draws: int = 0
    weyl_seed: Optional[int] = None

class SampleResponse(BaseModel):
    items: List[List[float]]  # [[index, value], ...]

class PotentialRequest(BaseModel):
    anchors: List[Anchor]
    kernel: str = "cosine"
    rbf_sigma: float = 0.8
    curiosity_alpha: float = 0.12
    curiosity_visits: Optional[List[float]] = None

class PotentialResponse(BaseModel):
    values: List[float]

class ConsolidateRequest(BaseModel):
    vectors: List[List[float]]

class ConsolidateResponse(BaseModel):
    vectors: List[List[float]]

class BlueprintResponse(BaseModel):
    points: List[Dict[str, float]]

# ----- Router Factory -----

def build_router(services: Any) -> APIRouter:
    """
    Build an APIRouter with handlers that use provided services container.
    Expected services:
      - config: core.config.AppConfig
      - console: object with .log(*args)
      - physics: physics.e8_lattice.E8Physics
      - qengine: physics.engines.QuantumEngine
      - cengine: physics.engines.ClassicalEngine
      - memory: memory.manager.MemoryManager
    """
    router = APIRouter()

    @router.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @router.get("/blueprint", response_model=BlueprintResponse)
    def blueprint(seed: Optional[int] = None):
        try:
            pts = services.cengine.blueprint(seed=seed)
            # FastAPI will serialize dicts; ensure float conversion
            return {"points": pts}
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"blueprint_error: {e}")

    @router.post("/sample", response_model=SampleResponse)
    def sample(req: SampleRequest):
        try:
            anchors = [(np.asarray(a.vector, dtype=np.float32), float(a.weight)) for a in req.anchors]
            if req.curiosity_visits is not None:
                visits = np.asarray(req.curiosity_visits, dtype=np.float32)
            else:
                visits = None
            services.qengine.set_anchors(anchors)
            services.qengine.potential(
                curiosity_visits=visits,
                curiosity_alpha=float(req.curiosity_alpha),
                weyl_draws=int(req.weyl_draws),
                weyl_seed=req.weyl_seed,
            )
            items = services.qengine.sample(top_k=int(req.top_k))
            return {"items": [[int(i), float(v)] for i, v in items]}
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"sample_error: {e}")

    @router.post("/potential", response_model=PotentialResponse)
    def potential(req: PotentialRequest):
        try:
            anchors = [(np.asarray(a.vector, dtype=np.float32), float(a.weight)) for a in req.anchors]
            visits = np.asarray(req.curiosity_visits, dtype=np.float32) if req.curiosity_visits is not None else None
            V = services.cengine.potential_from_anchors(
                anchors=anchors,
                kernel=req.kernel,
                rbf_sigma=float(req.rbf_sigma),
                curiosity_visits=visits,
                curiosity_alpha=float(req.curiosity_alpha),
            )
            return {"values": [float(x) for x in V.tolist()]}
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"potential_error: {e}")

    @router.post("/consolidate", response_model=ConsolidateResponse)
    def consolidate(req: ConsolidateRequest):
        try:
            X = np.asarray(req.vectors, dtype=np.float32)
            if X.ndim == 1:
                X = X.reshape(1, -1)
            if X.shape[1] != services.memory.embed_dim:
                raise HTTPException(status_code=400, detail=f"vectors must have dim={services.memory.embed_dim}")
            Y = services.memory.consolidate(X).astype(np.float32)
            return {"vectors": Y.tolist()}
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"consolidate_error: {e}")

    return router
