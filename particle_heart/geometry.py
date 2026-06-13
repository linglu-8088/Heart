from __future__ import annotations

import numpy as np

HEART_SCALE = 1.0 / 22.0
HEART_DEPTH = 0.72


def heart_boundary(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classic 2D parametric heart curve."""
    x = 17.5 * np.sin(theta) ** 3
    y = (
        13.0 * np.cos(theta)
        - 5.0 * np.cos(2.0 * theta)
        - 2.0 * np.cos(3.0 * theta)
        - np.cos(4.0 * theta)
    )
    return x * HEART_SCALE, y * HEART_SCALE


def _heart_outward_normal(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-5
    x0, y0 = heart_boundary(theta - eps)
    x1, y1 = heart_boundary(theta + eps)
    tx = x1 - x0
    ty = y1 - y0
    n_len = np.sqrt(tx * tx + ty * ty)
    n_len = np.maximum(n_len, 1e-8)
    return ty / n_len, -tx / n_len


def _heart_local_radius(theta):
    bx, by = heart_boundary(theta)
    return np.sqrt(bx * bx + by * by)


def _z_depth(r_factor, theta, count, rng):
    local_r = _heart_local_radius(theta)
    z_scale = np.clip(local_r / 1.25, 0.30, 1.0)
    base = HEART_DEPTH * (
        0.22 + 0.78 * np.power(np.maximum(0.0, 1.0 - r_factor), 0.55)
    )
    z_max = base * z_scale
    z_gauss = rng.normal(0.0, z_max * 0.52, count)
    z_extra = (
        (rng.random(count) * 2.0 - 1.0)
        * np.clip(1.0 - r_factor * 0.8, 0.15, 1.0)
        * 0.15
    )
    return z_gauss + z_extra
