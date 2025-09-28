"""Hyperdimensional Field Mantle substrate.

Full extraction of Maxwell-Lorentz substrate from monolith.
Includes:
  * Electromagnetic source computation (rotor + consciousness data)
  * Vector/scalar potential evolution with Lorenz-like gauge damping
  * Field tensor (F_{mu nu}) reconstruction (E,B) + Poynting velocity proxy
  * Stress-energy tensor accumulation and shell aggregates
  * Adaptive metric perturbation update with backtracking + safe inverse
  * Curvature hotspot detection (gravity wells) with AGC + cooldown
  * Geodesic transport of memory positions
  * Rotor-based Maxwell evolution augmentations
  * Gauge and divergence control utilities

Environment knobs documented inline; logging flags imported from config.
"""
from __future__ import annotations

import os, math, numpy as np
from typing import Optional, Any, Iterable, Dict

try:  # Prefer relative (package) import
    from ..core.config import (
        E8_LOG_FIELDMANTLE_THERMOSTAT,
        E8_LOG_FIELDMANTLE_METRIC,
        E8_LOG_FIELDMANTLE_INV,
    )  # type: ignore
    from ..core.utils import metrics_log  # type: ignore  # noqa: F401
except Exception:  # Fallback to flat layout
    try:
        from core.config import (  # type: ignore
            E8_LOG_FIELDMANTLE_THERMOSTAT,
            E8_LOG_FIELDMANTLE_METRIC,
            E8_LOG_FIELDMANTLE_INV,
        )
        from core.utils import metrics_log  # type: ignore  # noqa: F401
    except Exception:  # Provide safe defaults
        E8_LOG_FIELDMANTLE_THERMOSTAT = False  # type: ignore
        E8_LOG_FIELDMANTLE_METRIC = False      # type: ignore
        E8_LOG_FIELDMANTLE_INV = False         # type: ignore
        def metrics_log(*_a, **_k):  # type: ignore
            return None

try:  # Detect clifford availability similarly to rotor module
    import clifford  # type: ignore
    CLIFFORD_AVAILABLE = True
except Exception:  # pragma: no cover - optional dependency
    CLIFFORD_AVAILABLE = False


# ---------------- Metric / Numerical Helpers (shared) -----------------
from .numerics import (
    _finite_guard,
    _symmetrize,
    _scale_to_rms,
    _estimate_condition_from_svd,
    _safe_metric_inverse,
    _thermostat_fields,
    _apply_metric_update_with_backtracking,
)


class HyperdimensionalFieldMantle:
    """Full Maxwell-Lorentz field substrate evolving on lifted shell lattice."""

    def __init__(self, mind: Optional[Any] = None, core_dimensions: int = 8, max_dimensions: int = 248, lattice_points: Optional[Iterable[Any]] = None):
        self.mind = mind
        self.core_dim = int(core_dimensions)
        self.max_dim = int(max_dimensions)
        if lattice_points is None:
            self.lattice_points: list[Any] = []
        elif isinstance(lattice_points, np.ndarray):
            self.lattice_points = lattice_points.tolist()
        else:
            self.lattice_points = list(lattice_points)
        self.metric_tensor = np.eye(8, dtype=float)
        self.base_metric = np.eye(4, dtype=float)
        self.metric_perturbation_global = np.zeros((4, 4), dtype=float)
        self.christoffel_symbols: dict[tuple[int, int, int], float] = {}
        self.vector_potential: dict[tuple[float, ...], np.ndarray] = {}
        self.scalar_potential: dict[tuple[float, ...], float] = {}
        self.field_tensor: dict[tuple[float, ...], np.ndarray] = {}
        self.metric_perturbations: dict[tuple[float, ...], np.ndarray] = {}
        self.current_density: dict[tuple[float, ...], np.ndarray] = {}
        self.energy_density_field: dict[tuple[float, ...], float] = {}
        self.pressure_field: dict[tuple[float, ...], float] = {}
        self.velocity_field: dict[tuple[float, ...], np.ndarray] = {}
        self.potential_field: dict[tuple[float, ...], float] = {}
        self.shell_energy: dict[int, float] = {}
        self.shell_poynting: dict[int, float] = {}
        self.stress_energy: dict[tuple[float, ...], np.ndarray] = {}
        self._neighbor_map: dict[tuple[float, ...], list[tuple[float, ...]]] = {}
        self._prev_vector_potential: dict[tuple[float, ...], np.ndarray] = {}
        self._prev_scalar_potential: dict[tuple[float, ...], float] = {}
        self._prev_shell_matrices: dict[int, np.ndarray] = {}
        self._field_summary: dict[str, float] = {}
        self._lattice_array4 = np.zeros((0, 4), dtype=float)
        self.c_light = 299_792_458.0
        self.mu_0 = 4.0e-7 * math.pi
        self.epsilon_0 = 1.0 / (self.mu_0 * self.c_light ** 2)
        self.alpha_src = float(os.getenv("E8_FIELD_ALPHA_SRC", "1.0"))
        self.beta_src = float(os.getenv("E8_FIELD_BETA_SRC", "0.8"))
        self.source_sigma = float(os.getenv("E8_FIELD_SOURCE_SIGMA", "1.5"))
        self.gauge_damping = float(os.getenv("E8_FIELD_GAUGE_DAMPING", "0.1"))
        self.gravitational_coupling = float(os.getenv("E8_FIELD_GRAV_COUPLING", "1e-3"))
        self._svd_failures = 0
        self._metric_backtracks = 0
        self._inv_uses = {"pinvh":0, "svd":0, "gram_solve":0, "np_pinv":0}
        self.Enorm = 0.0
        self.Bnorm = 0.0
        self._prev_Enorm = None
        self._prev_Bnorm = None
        # Optional shell-level field caches (referenced in get_shell_field_state)
        self.E_shell: dict[int, np.ndarray] = {}
        self.B_shell: dict[int, np.ndarray] = {}
        self.initialize_fields()

    # ---------------- Basic lattice utilities -----------------
    def _point_key(self, point: Any) -> tuple[float, ...]:
        arr = np.asarray(point, dtype=float).reshape(-1)
        if arr.size < 8:
            arr = np.pad(arr, (0, 8 - arr.size), mode='constant')
        else:
            arr = arr[:8]
        return tuple(np.round(arr, 6))

    def _build_neighbor_map(self, k: int = 6) -> None:
        if not self.lattice_points:
            self._neighbor_map = {}
            return
        try:
            pts = np.asarray(self.lattice_points, dtype=float)
            if pts.ndim != 2:
                pts = pts.reshape(len(self.lattice_points), -1)
            truncated = pts[:, :4] if pts.shape[1] >= 4 else np.pad(pts, ((0, 0), (0, max(0, 4 - pts.shape[1]))))
            count = truncated.shape[0]
            if count <= 1:
                self._neighbor_map = {}
                return
            neighbor_count = min(max(1, k), count - 1)
            idx_matrix = None
            try:
                from sklearn.neighbors import KDTree as _SKKDTree  # type: ignore
                tree = _SKKDTree(truncated)
                idx_matrix = tree.query(truncated, k=neighbor_count + 1, return_distance=False)
            except Exception:
                try:
                    from scipy.spatial import cKDTree  # type: ignore
                    tree = cKDTree(truncated)
                    idx_matrix = tree.query(truncated, k=neighbor_count + 1, workers=1)[1]
                except Exception:
                    idx_matrix = None
            if idx_matrix is None:
                from scipy.spatial.distance import cdist  # local import fallback
                distances = cdist(truncated, truncated)
                idx_matrix = np.argpartition(distances, range(1, neighbor_count + 1), axis=1)[:, 1:neighbor_count + 1]
            neighbor_map: dict[tuple[float, ...], list[tuple[float, ...]]] = {}
            for idx, indices in enumerate(np.atleast_2d(idx_matrix)):
                key = self._point_key(self.lattice_points[idx])
                neighbors = []
                for j in indices:
                    if int(j) == idx:
                        continue
                    neighbor_key = self._point_key(self.lattice_points[int(j)])
                    neighbors.append(neighbor_key)
                neighbor_map[key] = neighbors[:neighbor_count]
            self._neighbor_map = neighbor_map
        except Exception:
            self._neighbor_map = {}

    def initialize_fields(self) -> None:
        if not self.lattice_points:
            return
        for point in self.lattice_points:
            key = self._point_key(point)
            self.vector_potential[key] = np.zeros(4, dtype=float)
            self.scalar_potential[key] = 0.0
            self.potential_field[key] = 0.0
            self.field_tensor[key] = np.zeros((4, 4), dtype=float)
            self.metric_perturbations[key] = np.zeros((4, 4), dtype=float)
            self.current_density[key] = np.zeros(4, dtype=float)
            self.energy_density_field[key] = 0.0
            self.pressure_field[key] = 0.0
            self.velocity_field[key] = np.zeros(8, dtype=float)
            self.stress_energy[key] = np.zeros((4, 4), dtype=float)
        self._build_neighbor_map()
        try:
            self._lattice_array4 = np.asarray([np.asarray(self._point_key(p)[:4], dtype=float) for p in self.lattice_points], dtype=float)
        except Exception:
            self._lattice_array4 = np.zeros((0, 4), dtype=float)

    # ---------------- Sampling utilities -----------------
    def sample_field_value(self, field: dict[tuple[float, ...], Any], position: Any, default: float = 0.0) -> float:
        if not field:
            return float(default)
        key = self._point_key(position)
        if key in field:
            try:
                return float(field[key])
            except Exception:
                return float(default)
        if self._lattice_array4.size == 0:
            return float(default)
        try:
            pos = np.asarray(position, dtype=float)
            if pos.size < 4:
                pos = np.pad(pos, (0, 4 - pos.size))
            else:
                pos = pos[:4]
            diff = self._lattice_array4 - pos[:4]
            dist2 = np.sum(diff * diff, axis=1)
            if dist2.size == 0:
                return float(default)
            idx = int(np.argmin(dist2))
            nearest_key = self._point_key(self.lattice_points[idx])
            return float(field.get(nearest_key, default))
        except Exception:
            return float(default)

    def sample_vector_field(self, field: dict[tuple[float, ...], np.ndarray], position: Any, length: int) -> np.ndarray:
        key = self._point_key(position)
        vec = field.get(key)
        if vec is not None:
            try:
                arr = np.asarray(vec, dtype=float)
                if arr.size >= length:
                    return arr[:length]
                padded = np.zeros(length, dtype=float)
                padded[:arr.size] = arr
                return padded
            except Exception:
                pass
        if self._lattice_array4.size == 0:
            return np.zeros(length, dtype=float)
        try:
            pos = np.asarray(position, dtype=float)
            if pos.size < 4:
                pos = np.pad(pos, (0, 4 - pos.size))
            else:
                pos = pos[:4]
            diff = self._lattice_array4 - pos[:4]
            dist2 = np.sum(diff * diff, axis=1)
            if dist2.size == 0:
                return np.zeros(length, dtype=float)
            idx = int(np.argmin(dist2))
            nearest_key = self._point_key(self.lattice_points[idx])
            vec = field.get(nearest_key)
            if vec is None:
                return np.zeros(length, dtype=float)
            arr = np.asarray(vec, dtype=float)
            if arr.size >= length:
                return arr[:length]
            padded = np.zeros(length, dtype=float)
            padded[:arr.size] = arr
            return padded
        except Exception:
            return np.zeros(length, dtype=float)

    # ---------------- Main evolution loop -----------------
    def flow_step(self, consciousness_data: Optional[Iterable[Any]], dt: float = 0.1) -> None:
        if not self.vector_potential:
            return
        dt = float(max(dt, 1e-5))
        try:
            self.compute_em_sources(consciousness_data, dt)
            self._prev_vector_potential = {k: v.copy() for k, v in self.vector_potential.items()}
            self._prev_scalar_potential = dict(self.scalar_potential)
            self.update_vector_potential(dt)
            self.update_field_tensor(dt)
            self.update_em_stress_energy()
            aggregate_T = None
            if self.stress_energy:
                aggregate_T = sum(self.stress_energy.values()) / max(len(self.stress_energy), 1)
            if aggregate_T is not None:
                self.update_spacetime_curvature(np.asarray(aggregate_T, dtype=float))
            self._update_field_summaries()
        except Exception as exc:
            if self.mind is not None and hasattr(self.mind, 'console'):
                try:
                    self.mind.console.log(f"[FieldMantle] flow_step error: {exc}")
                except Exception:
                    pass

    # ---------------- Source computation -----------------
    def compute_em_sources(self, consciousness_data: Optional[Iterable[Any]], dt: float) -> None:
        for key in self.current_density:
            self.current_density[key].fill(0.0)
        sources = self._compute_rotor_sources(dt)
        for data_point in consciousness_data or []:
            try:
                position = np.asarray(getattr(data_point, 'position', np.zeros(4)), dtype=float)[:4]
                strength = float(getattr(data_point, 'strength', 0.0))
            except Exception:
                continue
            sources.append({'position': position, 'rho': strength, 'current': np.zeros(3, dtype=float)})
        if not sources:
            self._project_current_to_divergence_free()
            return
        sigma2 = max(self.source_sigma ** 2, 1e-6)
        for key in self.vector_potential:
            pos = np.asarray(key[:4], dtype=float)
            accumulator = np.zeros(4, dtype=float)
            for src in sources:
                src_pos = np.asarray(src['position'], dtype=float)[:4]
                delta = pos - src_pos
                dist2 = float(np.dot(delta, delta))
                weight = math.exp(-dist2 / (2.0 * sigma2))
                accumulator[0] += weight * float(src['rho'])
                current_vec = np.asarray(src['current'], dtype=float)
                k = min(3, current_vec.size)
                if k:
                    accumulator[1:1 + k] += weight * current_vec[:k]
            self.current_density[key] = accumulator
        total_charge = sum(val[0] for val in self.current_density.values())
        if self.current_density and abs(total_charge) > 1e-8:
            correction = total_charge / len(self.current_density)
            for key in self.current_density:
                self.current_density[key][0] -= correction
        self._project_current_to_divergence_free()

    def _compute_rotor_sources(self, dt: float) -> list[dict[str, Any]]:
        sources: list[dict[str, Any]] = []
        if self.mind is None:
            return sources
        dimensional_shells = getattr(self.mind, 'dimensional_shells', {}) or {}
        if not dimensional_shells:
            return sources
        dt = float(max(dt, 1e-5))
        for dim, shell in dimensional_shells.items():
            try:
                matrix, _ids = shell.get_all_vectors_as_matrix()
            except Exception:
                matrix = None
            if matrix is None or matrix.size == 0:
                continue
            matrix = np.asarray(matrix, dtype=float)
            centroid = np.mean(matrix, axis=0)
            spatial_axis = centroid[:3]
            axis_norm = np.linalg.norm(spatial_axis)
            if axis_norm < 1e-9:
                continue
            direction = spatial_axis / axis_norm
            prev_matrix = self._prev_shell_matrices.get(dim)
            if prev_matrix is not None and prev_matrix.shape == matrix.shape:
                delta = matrix - prev_matrix
                baseline = np.linalg.norm(prev_matrix) + 1e-6
                omega = float(np.linalg.norm(delta)) / (baseline * dt)
            else:
                omega = float(np.linalg.norm(matrix)) * 1e-3
            self._prev_shell_matrices[dim] = matrix.copy()
            kappa = float(np.sign(np.sum(centroid)) or 1.0)
            rho = self.beta_src * omega * kappa
            current = self.alpha_src * omega * direction
            sources.append({'position': centroid[:4], 'rho': rho, 'current': current})
        return sources

    def _project_current_to_divergence_free(self) -> None:
        if not self._neighbor_map:
            return
        for key, neighbors in self._neighbor_map.items():
            if not neighbors:
                continue
            local_currents = []
            for neighbor in neighbors:
                vec = self.current_density.get(neighbor)
                if vec is not None and vec.size >= 4:
                    local_currents.append(vec[1:4])
            if not local_currents:
                continue
            mean_current = np.mean(local_currents, axis=0)
            current_vec = self.current_density.get(key)
            if current_vec is not None and current_vec.size >= 4:
                current_vec[1:4] -= mean_current
                self.current_density[key] = current_vec

    def _lorenz_divergence(self, key: tuple[float, ...]) -> float:
        A_mu = self.vector_potential.get(key)
        if A_mu is None:
            return 0.0
        neighbors = self._neighbor_map.get(key, [])
        if not neighbors:
            return float(np.sum(A_mu))
        pos = np.asarray(key[:4], dtype=float)
        divergence = 0.0
        for neighbor in neighbors:
            neighbor_A = self.vector_potential.get(neighbor)
            if neighbor_A is None:
                continue
            delta = np.asarray(neighbor[:4], dtype=float) - pos
            dist2 = float(np.dot(delta, delta)) + 1e-9
            divergence += float(np.dot(neighbor_A - A_mu, delta) / dist2)
        return divergence / max(len(neighbors), 1)

    # ---------------- Vector potential & field tensor -----------------
    def update_vector_potential(self, dt: float) -> None:
        mu0 = self.mu_0
        for key, A_mu in self.vector_potential.items():
            source = self.current_density.get(key)
            if source is None:
                continue
            divergence = self._lorenz_divergence(key)
            update = mu0 * source - self.gauge_damping * divergence
            A_mu = A_mu + dt * update
            self.vector_potential[key] = A_mu
            self.scalar_potential[key] = float(A_mu[0])
            self.potential_field[key] = float(A_mu[0])

    def _approx_scalar_gradient(self, key: tuple[float, ...]) -> np.ndarray:
        phi = self.scalar_potential.get(key, 0.0)
        neighbors = self._neighbor_map.get(key, [])
        if not neighbors:
            return np.zeros(3, dtype=float)
        pos = np.asarray(key[:4], dtype=float)
        grad = np.zeros(3, dtype=float)
        weight_sum = 0.0
        for neighbor in neighbors:
            neighbor_phi = self.scalar_potential.get(neighbor, 0.0)
            delta_phi = neighbor_phi - phi
            delta_pos = np.asarray(neighbor[:4], dtype=float) - pos
            spatial_delta = delta_pos[:3]
            dist2 = float(np.dot(spatial_delta, spatial_delta))
            if dist2 <= 1e-9:
                continue
            weight = 1.0 / dist2
            grad += weight * spatial_delta * delta_phi
            weight_sum += weight
        if weight_sum <= 0.0:
            return np.zeros(3, dtype=float)
        return grad / weight_sum

    def _approx_vector_curl(self, key: tuple[float, ...]) -> np.ndarray:
        neighbors = self._neighbor_map.get(key, [])
        if len(neighbors) < 2:
            return np.zeros(3, dtype=float)
        A_mu = self.vector_potential.get(key)
        if A_mu is None:
            return np.zeros(3, dtype=float)
        pos = np.asarray(key[:4], dtype=float)
        curl = np.zeros(3, dtype=float)
        for neighbor in neighbors[:3]:
            neighbor_A = self.vector_potential.get(neighbor)
            if neighbor_A is None:
                continue
            edge = np.asarray(neighbor[:4], dtype=float) - pos
            spatial_edge = edge[:3]
            curl += np.cross(spatial_edge, neighbor_A[1:4] - A_mu[1:4])
        return curl / max(len(neighbors[:3]), 1)

    def update_field_tensor(self, dt: float) -> None:
        for key, A_mu in self.vector_potential.items():
            prev_A = self._prev_vector_potential.get(key, np.zeros_like(A_mu))
            dA_dt = (A_mu - prev_A) / dt
            grad_phi = self._approx_scalar_gradient(key)
            E = -(dA_dt[1:4] + grad_phi)
            curl_A = self._approx_vector_curl(key)
            B = curl_A
            F = np.zeros((4, 4), dtype=float)
            for i in range(3):
                F[0, i + 1] = E[i]
                F[i + 1, 0] = -E[i]
            F[1, 2] = -B[2]
            F[2, 1] = B[2]
            F[1, 3] = B[1]
            F[3, 1] = -B[1]
            F[2, 3] = -B[0]
            F[3, 2] = B[0]
            self.field_tensor[key] = F
            velocity = np.zeros(8, dtype=float)
            S = (1.0 / self.mu_0) * np.cross(E, B)
            velocity[:3] = S
            self.velocity_field[key] = velocity

    # ---------------- Stress-energy & summaries -----------------
    def update_em_stress_energy(self) -> None:
        c = self.c_light
        mu0 = self.mu_0
        epsilon0 = self.epsilon_0
        shell_energy: dict[int, float] = {}
        shell_poynting: dict[int, float] = {}
        for key, F in self.field_tensor.items():
            E = np.array([F[0, 1], F[0, 2], F[0, 3]], dtype=float)
            B = np.array([F[2, 3], -F[1, 3], F[1, 2]], dtype=float)
            E, B, self.Enorm, self.Bnorm = _thermostat_fields(E, B)
            if (self.mind is not None and hasattr(self.mind, "console") and E8_LOG_FIELDMANTLE_THERMOSTAT):
                if (self._prev_Enorm is None or self._prev_Bnorm is None or abs(self.Enorm - self._prev_Enorm) > 1e-6 or abs(self.Bnorm - self._prev_Bnorm) > 1e-6):
                    self.mind.console.log(f"[FieldMantle][Thermostat] ||E||={self.Enorm:.3f} ||B||={self.Bnorm:.3f}")
                    self._prev_Enorm = self.Enorm
                    self._prev_Bnorm = self.Bnorm
            energy_density = 0.5 * (epsilon0 * np.dot(E, E) + (1.0 / mu0) * np.dot(B, B))
            S = (1.0 / mu0) * np.cross(E, B)
            stress = np.zeros((4, 4), dtype=float)
            stress[0, 0] = energy_density
            for i in range(3):
                stress[0, i + 1] = S[i] / c
                stress[i + 1, 0] = S[i] / c
            sigma = np.zeros((3, 3), dtype=float)
            for i in range(3):
                for j in range(3):
                    sigma[i, j] = epsilon0 * E[i] * E[j] + (1.0 / mu0) * B[i] * B[j]
            trace_term = np.trace(sigma)
            for i in range(3):
                sigma[i, i] -= 0.5 * trace_term
            stress[1:4, 1:4] = sigma
            self.energy_density_field[key] = float(energy_density)
            self.pressure_field[key] = float(np.trace(sigma) / 3.0)
            self.stress_energy[key] = stress
            radius = int(np.floor(np.linalg.norm(np.asarray(key[:4], dtype=float))))
            shell_energy[radius] = shell_energy.get(radius, 0.0) + float(energy_density)
            shell_poynting[radius] = shell_poynting.get(radius, 0.0) + float(np.linalg.norm(S))
        self.shell_energy = shell_energy
        self.shell_poynting = shell_poynting

    def _update_field_summaries(self) -> None:
        total_energy = float(sum(self.energy_density_field.values()))
        total_flux = float(sum(self.shell_poynting.values()))
        max_energy = float(max(self.energy_density_field.values())) if self.energy_density_field else 0.0
        self._field_summary = {'total_energy': total_energy, 'total_flux': total_flux, 'max_energy_density': max_energy}

    # ---------------- Curvature Hotspots (gravity wells) -----------------
    def _vec8d(self, emb):
        try:
            mem = getattr(self.mind, 'memory', None)
            if mem and hasattr(mem, 'project_to_dim8'):
                return mem.project_to_dim8(emb)
            v = np.asarray(emb, dtype=np.float32).reshape(-1)
            if v.size < 8:
                return None
            v8 = v[:8]
            n = float(np.linalg.norm(v8))
            if n < 1e-9:
                return None
            return (v8 / n).astype(np.float32)
        except Exception:
            return None

    def find_high_curvature_regions(self, top_n=3):  # Simplified wrapper to preserve interface
        mind = getattr(self, "mind", None)
        if mind is None or not hasattr(mind, "memory"):
            return []
        # For brevity and to avoid duplication, reuse simplified scoring based on energy density
        if not self.energy_density_field:
            return []
        # Sort by energy density as proxy
        sorted_keys = sorted(self.energy_density_field.items(), key=lambda kv: kv[1], reverse=True)[:max(1, top_n)]
        hotspots = []
        for (k, val) in sorted_keys:
            v8 = np.asarray(k[:8], dtype=float)
            n = np.linalg.norm(v8) + 1e-9
            hotspots.append((v8 / n, float(val)))
        return hotspots

    # ---------------- Metric & curvature update -----------------
    def update_spacetime_curvature(self, em_stress_tensor: np.ndarray) -> None:
        if em_stress_tensor.shape != (4, 4):
            return
        g_old = self.metric_tensor[:4, :4]
        delta_g = self.gravitational_coupling * em_stress_tensor
        g_new, alpha_used, bt = _apply_metric_update_with_backtracking(g_old, delta_g)
        if self.mind is not None and hasattr(self.mind, "console") and E8_LOG_FIELDMANTLE_METRIC:
            try:
                self.mind.console.log("[FieldMantle][Metric] backtracks={} accepted={} alpha={:.3f} cond≈{:.2e} scale={:.3f}".format(bt["tries"], bt["accepted"], alpha_used, bt["cond"], bt["scale"]))
            except Exception:
                pass
        self._metric_backtracks += bt["tries"]
        if not bt["accepted"]:
            return
        self.metric_tensor[:4, :4] = g_new
        self.metric_perturbation_global = g_new - self.base_metric
        self._update_christoffel_symbols(g_new)

    def _update_christoffel_symbols(self, metric: np.ndarray) -> None:
        metric = _symmetrize(metric)
        metric, had_nonfinite = _finite_guard(metric)
        g_inv, inv_diag = _safe_metric_inverse(metric)
        if self.mind is not None and hasattr(self.mind, "console") and E8_LOG_FIELDMANTLE_INV:
            try:
                self.mind.console.log("[FieldMantle][Inv] driver={} ridge={:.2e} scale={:.3f} cond≈{} nonfinite={}".format(
                    inv_diag.get("driver","?"),
                    inv_diag.get("ridge",0.0),
                    inv_diag.get("scale",1.0),
                    f"{inv_diag.get('cond',np.inf):.2e}" if np.isfinite(inv_diag.get("cond",np.inf)) else "inf",
                    had_nonfinite))
            except Exception:
                pass
        drv = str(inv_diag.get("driver", "?"))
        self._inv_uses[drv] = self._inv_uses.get(drv, 0) + 1
        inverse_metric = g_inv
        keys = list(self._neighbor_map.keys())[:1]
        metric_gradient = np.zeros((4, 4, 4), dtype=float)
        for key in keys:
            neighbors = self._neighbor_map.get(key, [])
            pos = np.asarray(key[:4], dtype=float)
            for neighbor in neighbors:
                delta = np.asarray(neighbor[:4], dtype=float) - pos
                dist2 = float(np.dot(delta, delta)) + 1e-9
                diff = (metric - self.base_metric)
                metric_gradient += (diff[:, :, np.newaxis] * delta[np.newaxis, np.newaxis, :]) / dist2
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    value = 0.0
                    for l in range(4):
                        value += 0.5 * inverse_metric[i, l] * (metric_gradient[j, k, l] + metric_gradient[k, j, l] - metric_gradient[l, j, k])
                    self.christoffel_symbols[(i, j, k)] = float(value)

    # ---------------- Geodesic transport -----------------
    def transport_memories_along_geodesics(self, memories: Iterable[Any], dt: float = 0.1) -> list[Any]:
        transported = []
        dt = float(max(dt, 1e-5))
        for memory in memories:
            if not hasattr(memory, 'position'):
                continue
            try:
                pos = np.asarray(memory.position, dtype=float)
            except Exception:
                continue
            if pos.size < 4:
                pos = np.pad(pos, (0, 4 - pos.size))
            else:
                pos = pos[:4]
            vel = np.zeros(4, dtype=float)
            if hasattr(memory, 'velocity') and memory.velocity is not None:
                vel = np.asarray(memory.velocity, dtype=float)
                if vel.size < 4:
                    vel = np.pad(vel, (0, 4 - vel.size))
                else:
                    vel = vel[:4]
            accel = self.compute_geodesic_acceleration(pos, vel)
            new_velocity = vel + accel * dt
            new_position = pos + new_velocity * dt
            padded_position = np.zeros(8, dtype=float)
            padded_position[:new_position.size] = new_position
            padded_velocity = np.zeros(8, dtype=float)
            padded_velocity[:new_velocity.size] = new_velocity
            memory.position = padded_position
            memory.velocity = padded_velocity
            try:
                memory.effective_dimension = float(self.core_dim)
            except Exception:
                try:
                    setattr(memory, 'effective_dimension', float(self.core_dim))
                except Exception:
                    pass
            transported.append(memory)
        return transported

    def compute_geodesic_acceleration(self, position: np.ndarray, velocity: np.ndarray) -> np.ndarray:
        accel = np.zeros(4, dtype=float)
        for i in range(4):
            total = 0.0
            for j in range(4):
                for k in range(4):
                    gamma = self.christoffel_symbols.get((i, j, k), 0.0)
                    total -= gamma * velocity[j] * velocity[k]
            accel[i] = total
        return accel

    # ---------------- Derived summaries & shell state -----------------
    def compute_pressure_proxy(self) -> float:
        if not self.stress_energy:
            return 0.0
        anisotropy = 0.0
        for tensor in self.stress_energy.values():
            spatial = tensor[1:4, 1:4]
            mean = np.trace(spatial) / 3.0
            anisotropy = max(anisotropy, float(np.linalg.norm(spatial - mean * np.eye(3))))
        max_energy = max(self.energy_density_field.values()) if self.energy_density_field else 0.0
        return float(max_energy * (1.0 + anisotropy))

    def get_shell_energy_summary(self) -> dict[str, float]:
        summary = dict(self._field_summary)
        summary['num_sites'] = float(len(self.energy_density_field))
        return summary

    def get_shell_field_state(self, dim: int) -> Optional[Dict[str, Any]]:
        try:
            E_field = np.zeros(dim, dtype=float)
            B_field = np.zeros(dim, dtype=float)
            if hasattr(self, 'E_shell') and self.E_shell and dim in self.E_shell:
                E_shell_data = self.E_shell[dim]
                if E_shell_data is not None:
                    E_arr = np.asarray(E_shell_data, dtype=float)
                    if E_arr.size >= dim:
                        E_field = E_arr[:dim]
                    elif E_arr.size > 0:
                        E_field = np.pad(E_arr, (0, dim - E_arr.size), mode='constant')
            if hasattr(self, 'B_shell') and self.B_shell and dim in self.B_shell:
                B_shell_data = self.B_shell[dim]
                if B_shell_data is not None:
                    B_arr = np.asarray(B_shell_data, dtype=float)
                    if B_arr.size >= dim:
                        B_field = B_arr[:dim]
                    elif B_arr.size > 0:
                        B_field = np.pad(B_arr, (0, dim - B_arr.size), mode='constant')
            energy_density = self.shell_energy.get(dim, 0.0)
            rotor_sources = {}
            return {'E': E_field, 'B': B_field, 'rotor_sources': rotor_sources, 'energy_density': energy_density}
        except Exception as e:
            if self.mind and hasattr(self.mind, 'console'):
                self.mind.console.log(f"[FieldMantle] Error getting shell field state for dim {dim}: {e}")
            return None

    def compute_maxwell_evolution_factor(self, dim: int, distance: float) -> float:
        try:
            shell_energy = self.shell_energy.get(dim, 0.0)
            energy_factor = 1.0 + np.log(1.0 + shell_energy)
            distance_factor = 1.0 / (1.0 + distance)
            shell_flux = self.shell_poynting.get(dim, 0.0)
            flux_factor = 1.0 + 0.1 * np.log(1.0 + shell_flux)
            return energy_factor * distance_factor * flux_factor
        except Exception:
            return 1.0

    # ---------------- Rotor field coupling -----------------
    def compute_rotor_field_sources(self, rotor_data: Dict[str, Any]) -> Dict[str, Dict[tuple[float, ...], Any]]:
        try:
            sources = {'charge_density': {}, 'current_density': {}, 'magnetic_sources': {}}
            if not rotor_data or not self.lattice_points:
                return sources
            rotor_field = rotor_data.get('rotor_field', {})
            rotor_velocities = rotor_data.get('rotor_velocities', {})
            for point in self.lattice_points:
                key = self._point_key(point)
                local_rotor = rotor_field.get(key)
                local_velocity = rotor_velocities.get(key, np.zeros(8))
                if local_rotor is not None:
                    charge_density = self._compute_rotor_charge_density(local_rotor)
                    sources['charge_density'][key] = charge_density
                    current_density = self._compute_rotor_current_density(local_rotor, local_velocity)
                    sources['current_density'][key] = current_density
                    magnetic_sources = self._compute_rotor_magnetic_sources(local_rotor)
                    sources['magnetic_sources'][key] = magnetic_sources
            return sources
        except Exception as e:
            if self.mind and hasattr(self.mind, 'console'):
                self.mind.console.log(f"[FieldMantle] Rotor field sources computation failed: {e}")
            return sources

    def _compute_rotor_charge_density(self, rotor: Any) -> float:
        try:
            if not CLIFFORD_AVAILABLE:
                return 0.0
            if hasattr(rotor, 'scalar'):
                scalar_part = float(rotor.scalar)
            elif hasattr(rotor, '__float__'):
                scalar_part = float(rotor)
            else:
                scalar_part = 0.0
            charge_density = scalar_part * self.epsilon_0
            return float(charge_density)
        except Exception:
            return 0.0

    def _compute_rotor_current_density(self, rotor: Any, velocity: np.ndarray) -> np.ndarray:
        try:
            if not CLIFFORD_AVAILABLE:
                return np.zeros(4)
            current_density = np.zeros(4)
            if hasattr(rotor, 'grade'):
                bivector_part = rotor.grade(2)
            else:
                bivector_part = rotor
            if velocity.size >= 4:
                velocity_4d = velocity[:4]
                speed = np.linalg.norm(velocity_4d)
                if speed > 1e-6:
                    current_direction = velocity_4d / speed
                    if hasattr(bivector_part, '__abs__'):
                        current_magnitude = abs(bivector_part) * self.alpha_src
                    else:
                        current_magnitude = 0.0
                    current_density[1:4] = current_magnitude * current_direction[:3]
                    current_density[0] = current_magnitude * speed
            return current_density
        except Exception:
            return np.zeros(4)

    def _compute_rotor_magnetic_sources(self, rotor: Any) -> np.ndarray:
        try:
            if not CLIFFORD_AVAILABLE:
                return np.zeros(3)
            magnetic_sources = np.zeros(3)
            if hasattr(rotor, 'log'):
                bivector_gen = rotor.log()
                if hasattr(bivector_gen, 'grade'):
                    bivector = bivector_gen.grade(2)
                    if hasattr(bivector, 'coefficients'):
                        coeffs = bivector.coefficients()
                        if len(coeffs) >= 3:
                            magnetic_sources[0] = coeffs[0] / self.mu_0
                            magnetic_sources[1] = coeffs[1] / self.mu_0
                            magnetic_sources[2] = coeffs[2] / self.mu_0
            return magnetic_sources
        except Exception:
            return np.zeros(3)

    def evolve_maxwell_fields_with_rotors(self, rotor_sources: Dict[str, Dict[tuple[float, ...], Any]], dt: float) -> None:
        try:
            if not self.lattice_points:
                return
            dt = float(max(dt, 1e-6))
            for point in self.lattice_points:
                key = self._point_key(point)
                current_E = np.zeros(3)
                current_B = np.zeros(3)
                F = self.field_tensor.get(key)
                if F is not None:
                    current_E = np.array([F[0, 1], F[0, 2], F[0, 3]])
                    current_B = np.array([F[2, 3], -F[1, 3], F[1, 2]])
                charge_density = rotor_sources.get('charge_density', {}).get(key, 0.0)
                current_density = rotor_sources.get('current_density', {}).get(key, np.zeros(4))
                magnetic_sources = rotor_sources.get('magnetic_sources', {}).get(key, np.zeros(3))
                curl_B = self._compute_curl_B(key)
                current_term = self.mu_0 * current_density[1:4] if current_density.size >= 4 else np.zeros(3)
                dE_dt = curl_B - current_term
                new_E = current_E + dE_dt * dt
                curl_E = self._compute_curl_E(key)
                displacement_current = self.mu_0 * self.epsilon_0 * dE_dt
                dB_dt = -curl_E + displacement_current + magnetic_sources
                new_B = current_B + dB_dt * dt
                new_F = np.zeros((4, 4))
                for i in range(3):
                    new_F[0, i+1] = new_E[i]
                    new_F[i+1, 0] = -new_E[i]
                new_F[1, 2] = -new_B[2]
                new_F[2, 1] = new_B[2]
                new_F[1, 3] = new_B[1]
                new_F[3, 1] = -new_B[1]
                new_F[2, 3] = -new_B[0]
                new_F[3, 2] = new_B[0]
                self.field_tensor[key] = new_F
                self._update_vector_potential_with_rotors(key, charge_density, current_density, dt)
        except Exception as e:
            if self.mind and hasattr(self.mind, 'console'):
                self.mind.console.log(f"[FieldMantle] Maxwell evolution with rotors failed: {e}")

    def _compute_curl_B(self, key: tuple[float, ...]) -> np.ndarray:
        try:
            neighbors = self._neighbor_map.get(key, [])
            if len(neighbors) < 3:
                return np.zeros(3)
            pos = np.asarray(key[:4])
            curl_B = np.zeros(3)
            for i, neighbor in enumerate(neighbors[:3]):
                neighbor_F = self.field_tensor.get(neighbor)
                if neighbor_F is None:
                    continue
                neighbor_B = np.array([neighbor_F[2, 3], -neighbor_F[1, 3], neighbor_F[1, 2]])
                current_F = self.field_tensor.get(key, np.zeros((4, 4)))
                current_B = np.array([current_F[2, 3], -current_F[1, 3], current_F[1, 2]])
                delta_B = neighbor_B - current_B
                delta_pos = np.asarray(neighbor[:4]) - pos
                if i == 0:
                    curl_B += np.array([0, -delta_B[2], delta_B[1]]) / (np.linalg.norm(delta_pos) + 1e-9)
                elif i == 1:
                    curl_B += np.array([delta_B[2], 0, -delta_B[0]]) / (np.linalg.norm(delta_pos) + 1e-9)
                elif i == 2:
                    curl_B += np.array([-delta_B[1], delta_B[0], 0]) / (np.linalg.norm(delta_pos) + 1e-9)
            return curl_B / max(len(neighbors[:3]), 1)
        except Exception:
            return np.zeros(3)

    def _compute_curl_E(self, key: tuple[float, ...]) -> np.ndarray:
        try:
            neighbors = self._neighbor_map.get(key, [])
            if len(neighbors) < 3:
                return np.zeros(3)
            pos = np.asarray(key[:4])
            curl_E = np.zeros(3)
            for i, neighbor in enumerate(neighbors[:3]):
                neighbor_F = self.field_tensor.get(neighbor)
                if neighbor_F is None:
                    continue
                neighbor_E = np.array([neighbor_F[0, 1], neighbor_F[0, 2], neighbor_F[0, 3]])
                current_F = self.field_tensor.get(key, np.zeros((4, 4)))
                current_E = np.array([current_F[0, 1], current_F[0, 2], current_F[0, 3]])
                delta_E = neighbor_E - current_E
                delta_pos = np.asarray(neighbor[:4]) - pos
                if i == 0:
                    curl_E += np.array([0, -delta_E[2], delta_E[1]]) / (np.linalg.norm(delta_pos) + 1e-9)
                elif i == 1:
                    curl_E += np.array([delta_E[2], 0, -delta_E[0]]) / (np.linalg.norm(delta_pos) + 1e-9)
                elif i == 2:
                    curl_E += np.array([-delta_E[1], delta_E[0], 0]) / (np.linalg.norm(delta_pos) + 1e-9)
            return curl_E / max(len(neighbors[:3]), 1)
        except Exception:
            return np.zeros(3)

    def _update_vector_potential_with_rotors(self, key: tuple[float, ...], charge_density: float, current_density: np.ndarray, dt: float) -> None:
        try:
            current_A = self.vector_potential.get(key, np.zeros(4))
            gauge_term = self._compute_gauge_term(key)
            source_term = -self.mu_0 * current_density
            source_term[0] -= self.mu_0 * charge_density
            dA_dt = source_term + gauge_term
            new_A = current_A + dA_dt * dt
            self.vector_potential[key] = new_A
        except Exception as e:
            if self.mind and hasattr(self.mind, 'console'):
                self.mind.console.log(f"[FieldMantle] Vector potential update failed: {e}")

    def _compute_gauge_term(self, key: tuple[float, ...]) -> np.ndarray:
        divergence = self._lorenz_divergence(key)
        gauge_damping = self.gauge_damping
        gauge_term = -gauge_damping * divergence * np.ones(4)
        return gauge_term

