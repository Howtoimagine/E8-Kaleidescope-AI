"""Shared numerical helpers for physics modules.

Centralizes robust matrix utilities and field safety transforms originally
embedded in the mantle module. Import from here to avoid duplication.
"""
from __future__ import annotations

import os
import numpy as np
from typing import Tuple, Dict


def _finite_guard(M: np.ndarray):
    bad = ~np.isfinite(M)
    if bad.any():
        M = M.copy()
        M[bad] = 0.0
        return M, True
    return M, False


def _symmetrize(M: np.ndarray):
    return 0.5 * (M + M.T)


def _scale_to_rms(M: np.ndarray, target_rms: float = 1.0):
    rms = float(np.sqrt(np.mean(M * M)) + 1e-18)
    return M * (target_rms / rms), rms


def _estimate_condition_from_svd(Ms: np.ndarray):
    try:
        s = np.linalg.svd(Ms, compute_uv=False)
        if s.size == 0:
            return np.inf, 0.0, 0.0
        return float(s[0] / max(s[-1], 1e-18)), float(s[0]), float(s[-1])
    except Exception:
        return np.inf, 0.0, 0.0


def _safe_metric_inverse(g: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
    """Robust inverse with multi-driver fallback.
    Returns (inverse_metric, diagnostics_dict).
    """
    g = np.asarray(g, dtype=np.float64)
    ridge_min = float(os.getenv("E8_FM_RIDGE_MIN", "1e-10"))
    ridge_max = float(os.getenv("E8_FM_RIDGE_MAX", "1e-3"))
    target_rms = float(os.getenv("E8_FM_METRIC_TARGET_RMS", "1.0"))
    diag = {"driver": "?", "ridge": 0.0, "scale": 1.0, "cond": np.inf, "had_nonfinite": False}
    g, had_nonfinite = _finite_guard(g)
    g = _symmetrize(g)
    g_s, scale = _scale_to_rms(g, target_rms=target_rms)
    I = np.eye(g_s.shape[0], dtype=g_s.dtype)
    # Attempt pinvh
    try:
        from scipy.linalg import pinvh  # type: ignore
        lam = ridge_min
        g_inv_s = pinvh(g_s + lam * I)
        g_inv_s = np.asarray(g_inv_s, dtype=float)
        cond, _smax, _smin = _estimate_condition_from_svd(g_s)
        diag.update(driver="pinvh", ridge=lam, scale=scale, cond=cond, had_nonfinite=had_nonfinite)
        return g_inv_s / scale, diag
    except Exception:
        pass
    # Attempt SVD manual
    try:
        U, s, Vt = np.linalg.svd(g_s, full_matrices=False)
        smax = float(s[0]) if s.size else 1.0
        smin = float(s[-1]) if s.size else 1e-18
        cond = smax / max(smin, 1e-18)
        lam = min(ridge_max, max(ridge_min, 1e-6 * cond))
        s_inv = 1.0 / (s + lam)
        g_inv_s = (Vt.T * s_inv) @ U.T
        g_inv_s = np.asarray(g_inv_s, dtype=float)
        diag.update(driver="svd", ridge=lam, scale=scale, cond=cond, had_nonfinite=had_nonfinite)
        return g_inv_s / scale, diag
    except Exception:
        pass
    # Attempt Gram solve
    try:
        GTG = g_s.T @ g_s + ridge_max * I
        g_inv_s = np.linalg.solve(GTG, g_s.T)
        g_inv_s = np.asarray(g_inv_s, dtype=float)
        diag.update(driver="gram_solve", ridge=ridge_max, scale=scale, cond=np.inf, had_nonfinite=had_nonfinite)
        return g_inv_s / scale, diag
    except Exception:
        pass
    # Final fallback: np.linalg.pinv
    g_fallback = g + ridge_max * np.eye(g.shape[0], dtype=g.dtype)
    diag.update(driver="np_pinv", ridge=ridge_max, scale=1.0, cond=np.inf, had_nonfinite=had_nonfinite)
    return np.linalg.pinv(g_fallback), diag


def _thermostat_fields(E: np.ndarray, B: np.ndarray):
    E = np.asarray(E, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    E_sat = float(os.getenv("E8_FM_E_SAT", "8.0"))
    B_sat = float(os.getenv("E8_FM_B_SAT", "8.0"))
    gain = float(os.getenv("E8_FM_TANH_GAIN", "0.75"))
    E_safe = E_sat * np.tanh(gain * (E / (E_sat + 1e-18)))
    B_safe = B_sat * np.tanh(gain * (B / (B_sat + 1e-18)))
    E_safe[~np.isfinite(E_safe)] = 0.0
    B_safe[~np.isfinite(B_safe)] = 0.0
    En = float(np.sqrt(np.mean(E_safe * E_safe)))
    Bn = float(np.sqrt(np.mean(B_safe * B_safe)))
    return E_safe, B_safe, En, Bn


def _apply_metric_update_with_backtracking(g_old: np.ndarray, delta_g: np.ndarray):
    cond_limit = float(os.getenv("E8_FM_COND_LIMIT", "1e12"))
    max_bt = int(os.getenv("E8_FM_BACKTRACKS", "5"))
    shrink = float(os.getenv("E8_FM_STEP_SHRINK", "0.5"))
    target_rms = float(os.getenv("E8_FM_METRIC_TARGET_RMS", "1.0"))
    alpha = 1.0
    for tries in range(max_bt + 1):
        g_new = _symmetrize(g_old + alpha * delta_g)
        g_new, had_nonfinite = _finite_guard(g_new)
        g_scaled, scale = _scale_to_rms(g_new, target_rms=target_rms)
        cond, smax, smin = _estimate_condition_from_svd(g_scaled)
        if (not had_nonfinite) and (cond < cond_limit):
            return g_new, alpha, {"cond": cond, "scale": scale, "tries": tries, "accepted": True}
        alpha *= shrink
    return g_old, 0.0, {"cond": np.inf, "scale": 1.0, "tries": max_bt, "accepted": False}


__all__ = [
    "_finite_guard",
    "_symmetrize",
    "_scale_to_rms",
    "_estimate_condition_from_svd",
    "_safe_metric_inverse",
    "_thermostat_fields",
    "_apply_metric_update_with_backtracking",
]
