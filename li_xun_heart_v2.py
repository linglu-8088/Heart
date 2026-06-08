from __future__ import annotations

import argparse
import math
import sys
import time

import glfw
import moderngl
import numpy as np

# ═══════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════

WINDOW_TITLE = "3D Particle Heart v2"
BACKGROUND = (0.015, 0.0, 0.025, 1.0)

# Default particle count ratios (applied to --particles total)
_BASE_STARS = 3000
_BASE_SPARKS = 5000
_BASE_RINGS = 2048


# Body self-rotation speed (rad/s, ~42s per full revolution)
_ROTATION_SPEED = 0.15

# 3D depth: max Z half-thickness at centre, floor at surface
_HEART_DEPTH = 0.72

# Color themes: each entry is a list of 5 vec3 tints (outline/shell/fill/core/orbit)
THEME_NAMES = ["Passion", "Aurora", "Sunset", "Ice", "Neon", "Gold"]
THEMES = [
    # 0: Passion (warm red-pink)
    [(1.00, 0.55, 0.60), (1.00, 0.50, 0.55), (1.00, 0.45, 0.50),
     (1.10, 0.50, 0.55), (1.00, 0.60, 0.50)],
    # 1: Aurora (cool blue-teal-purple)
    [(0.80, 0.85, 1.15), (0.75, 0.90, 1.20), (0.70, 0.95, 1.25),
     (0.65, 0.90, 1.30), (0.70, 0.85, 1.15)],
    # 2: Sunset (warm orange-gold-red)
    [(1.15, 0.90, 0.75), (1.20, 0.85, 0.70), (1.25, 0.80, 0.60),
     (1.30, 0.75, 0.55), (1.10, 0.95, 0.70)],
    # 3: Ice (cool cyan-blue-white)
    [(0.80, 1.00, 1.25), (0.75, 1.05, 1.30), (0.70, 1.10, 1.35),
     (0.65, 1.15, 1.40), (0.75, 1.00, 1.20)],
    # 4: Neon (vibrant neon pink-blue-purple)
    [(1.20, 0.60, 1.30), (0.60, 0.60, 1.35), (1.30, 0.50, 0.90),
     (0.70, 0.40, 1.40), (1.15, 0.55, 1.10)],
    # 5: Gold (champagne gold)
    [(1.20, 1.10, 0.70), (1.15, 1.05, 0.65), (1.25, 1.00, 0.55),
     (1.30, 0.90, 0.50), (1.10, 1.05, 0.65)],
]


# ═══════════════════════════════════════════════════════════════
# Section 1: 2D Parametric Heart Boundary
# ═══════════════════════════════════════════════════════════════


_HEART_SCALE = 1.0 / 22.0


def heart_boundary(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classic 2D parametric heart curve.
    Returns (x, y) in approx [-1.17, 1.17] x [-1.31, 0.89]."""
    x = 17.5 * np.sin(theta) ** 3
    y = (13.0 * np.cos(theta)
         - 5.0 * np.cos(2.0 * theta)
         - 2.0 * np.cos(3.0 * theta)
         - np.cos(4.0 * theta))
    return x * _HEART_SCALE, y * _HEART_SCALE


def _heart_outward_normal(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Approximate 2D outward normal of the heart boundary at each theta."""
    eps = 1e-5
    x0, y0 = heart_boundary(theta - eps)
    x1, y1 = heart_boundary(theta + eps)
    tx = x1 - x0
    ty = y1 - y0
    n_len = np.sqrt(tx * tx + ty * ty)
    n_len = np.maximum(n_len, 1e-8)
    return ty / n_len, -tx / n_len


def _heart_local_radius(theta):
    """Local radius of the 2D heart boundary at each theta."""
    bx, by = heart_boundary(theta)
    return np.sqrt(bx * bx + by * by)


def _z_depth(r_factor, theta, count, rng):
    """Z depth scaled by local heart radius — wide lobes get thick Z, narrow tip/cleft get thin Z.
    This makes the 3D shape heart-like from all rotation angles."""
    local_r = _heart_local_radius(theta)
    z_scale = np.clip(local_r / 1.25, 0.30, 1.0)
    base = _HEART_DEPTH * (0.22 + 0.78 * np.power(np.maximum(0.0, 1.0 - r_factor), 0.55))
    z_max = base * z_scale
    z_gauss = rng.normal(0.0, z_max * 0.52, count)
    # D1: additional edge Z spread — centre gets more thickness, edge tapers off
    # z_extra = (rand*2-1) * (1.0 - nd*0.8) where nd ≈ r_factor
    z_extra = (rng.random(count) * 2.0 - 1.0) * np.clip(1.0 - r_factor * 0.8, 0.15, 1.0) * 0.15
    return z_gauss + z_extra


# ═══════════════════════════════════════════════════════════════
# Section 3: Particle Generation
# ═══════════════════════════════════════════════════════════════

def _lerp_colors(ca: np.ndarray, cb: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t[:, None]
    return ca + (cb - ca) * t


def _gradient_3stop(t: np.ndarray, c0: np.ndarray, c1: np.ndarray, c2: np.ndarray,
                    t1: float = 0.45) -> np.ndarray:
    """H1: Three-stop colour gradient — 0→t1 interpolates c0→c1, t1→1 interpolates c1→c2."""
    result = np.zeros((t.shape[0], 3), dtype="f4")
    mask_lo = t < t1
    mask_hi = ~mask_lo
    if mask_lo.any():
        result[mask_lo] = c0 + (c1 - c0) * (t[mask_lo] / t1)[:, None]
    if mask_hi.any():
        result[mask_hi] = c1 + (c2 - c1) * ((t[mask_hi] - t1) / (1.0 - t1))[:, None]
    return result


def _stack_particles(positions, colors, sizes, phase, kind, distance, alpha, speed):
    kind_col = np.full((positions.shape[0], 1), kind, dtype="f4")
    return np.hstack([
        positions.astype("f4"),
        colors.astype("f4"),
        sizes[:, None].astype("f4"),
        phase[:, None].astype("f4"),
        kind_col,
        distance[:, None].astype("f4"),
        alpha[:, None].astype("f4"),
        speed[:, None].astype("f4"),
    ])


def _sample_outline_new(rng, count):
    # A1: Adaptive theta sampling — 72% uniform + 28% concentrated near bottom tip
    n_uniform = int(count * 0.72)
    n_tip = count - n_uniform
    theta_u = rng.uniform(0.0, math.tau, n_uniform)
    # Concentrate extra particles around theta=pi (bottom tip) using beta on [0.7pi, 1.3pi]
    tip_raw = rng.beta(3.5, 3.5, n_tip)  # peak at 0.5
    theta_tip = 0.7 * math.pi + (1.3 * math.pi - 0.7 * math.pi) * tip_raw
    theta = np.concatenate([theta_u, theta_tip])
    theta += rng.normal(0.0, 0.008, count)  # tightened from 0.010
    base_x, base_y = heart_boundary(theta)
    shell = 0.985 + 0.020 * rng.random(count) + rng.normal(0.0, 0.006, count)  # tightened shell range
    jitter = rng.normal(0.0, 0.010, (count, 2))
    x = base_x * shell + jitter[:, 0]
    y = base_y * shell + jitter[:, 1]

    r_factor = np.clip(0.82 + rng.random(count) * 0.22, 0.78, 1.04)
    z = _z_depth(r_factor, theta, count, rng)

    positions = np.column_stack([x, y, z])
    n = count
    # H1: 3-stop gradient: deep red #c2185b → bright red #ff2d55 → light gold
    deep_red = np.array([0.76, 0.09, 0.36], dtype="f4")
    bright_red = np.array([1.00, 0.18, 0.33], dtype="f4")
    light_gold = np.array([1.00, 0.72, 0.40], dtype="f4")
    nr = np.clip((r_factor - 0.78) / 0.26, 0.0, 1.0)
    nr += rng.normal(0.0, 0.10, n)
    nr = np.clip(nr, 0.0, 1.0)
    colors = _gradient_3stop(1.0 - nr, deep_red, bright_red, light_gold, 0.35)
    # C1: size tied to r_factor — centre larger, edge smaller
    size_factor = 0.60 + 0.40 * (1.0 - (r_factor - 0.78) / 0.26)
    sizes = rng.uniform(2.0, 4.0, n) * size_factor  # tightened range
    phase = rng.uniform(0.0, math.tau, n)
    # C2: edge particles slightly more transparent
    base_alpha = rng.uniform(0.66, 0.92, n)  # raised alpha baseline
    alpha_factor = 0.72 + 0.28 * (1.0 - (r_factor - 0.78) / 0.26)
    alpha = base_alpha * alpha_factor
    speed = rng.uniform(0.8, 1.3, n)
    return _stack_particles(positions, colors, sizes, phase, 0.0, r_factor, alpha, speed)


def _sample_loose_outline(rng, count):
    """F2: Loose outer halo — sits just outside the tight outline, softer and more transparent,
    creating a natural 'particle gathering' transition glow around the heart."""
    theta = rng.uniform(0.0, math.tau, count)
    theta += rng.normal(0.0, 0.020, count)
    base_x, base_y = heart_boundary(theta)
    # Sits just outside the tight outline: shell 1.015–1.045 (tightened)
    shell = 1.015 + 0.030 * rng.random(count) + rng.normal(0.0, 0.010, count)
    jitter = rng.normal(0.0, 0.012, (count, 2))  # reduced from 0.018
    x = base_x * shell + jitter[:, 0]
    y = base_y * shell + jitter[:, 1]

    r_factor = np.clip(0.88 + rng.random(count) * 0.16, 0.85, 1.08)
    z = rng.normal(0.0, _HEART_DEPTH * 0.10, count)  # reduced from 0.15

    positions = np.column_stack([x, y, z])
    n = count
    # Softer, more transparent colours — deep red to dark crimson
    dr = np.array([0.72, 0.08, 0.32], dtype="f4")  # deeper edge red
    cr = np.array([0.50, 0.04, 0.22], dtype="f4")  # dark crimson
    nr = np.clip(rng.normal(0.4, 0.25, n), 0.0, 1.0)
    colors = _lerp_colors(dr, cr, nr)
    sizes = rng.uniform(1.2, 3.0, n)
    phase = rng.uniform(0.0, math.tau, n)
    alpha = rng.uniform(0.06, 0.18, n)  # much more transparent, reduced from 0.15~0.35
    speed = rng.uniform(0.6, 1.1, n)
    return _stack_particles(positions, colors, sizes, phase, 0.3, r_factor, alpha, speed)


def _sample_shell_new(rng, count):
    theta = rng.uniform(0.0, math.tau, count)
    base_x, base_y = heart_boundary(theta)
    r_factor = 0.55 + 0.40 * (rng.random(count) ** 0.55)
    xy_noise = rng.normal(0.0, 0.025, (count, 2))
    x = base_x * r_factor + xy_noise[:, 0]
    y = base_y * r_factor + xy_noise[:, 1]
    z = _z_depth(r_factor, theta, count, rng)

    positions = np.column_stack([x, y, z])
    n = count
    # H1: 3-stop gradient: deep red → bright red → light gold
    deep_red = np.array([0.78, 0.10, 0.34], dtype="f4")
    bright_red = np.array([1.00, 0.22, 0.37], dtype="f4")
    light_gold = np.array([1.00, 0.72, 0.40], dtype="f4")
    nr = np.clip(r_factor / 0.95, 0.0, 1.0)
    nr += rng.normal(0.0, 0.10, n)
    nr = np.clip(nr, 0.0, 1.0)
    colors = _gradient_3stop(1.0 - nr, deep_red, bright_red, light_gold, 0.38)
    # C1: size tied to r_factor — centre larger, edge smaller
    size_factor = 0.70 + 0.30 * (1.0 - r_factor / 0.95)
    sizes = rng.uniform(2.0, 4.5, n) * size_factor
    phase = rng.uniform(0.0, math.tau, n)
    # C2: edge particles slightly more transparent
    base_alpha = rng.uniform(0.42, 0.72, n)  # raised
    alpha_factor = 0.72 + 0.28 * (1.0 - r_factor / 0.95)
    alpha = base_alpha * alpha_factor
    speed = rng.uniform(0.9, 1.6, n)
    return _stack_particles(positions, colors, sizes, phase, 1.0, r_factor, alpha, speed)


def _sample_fill_new(rng, count):
    # F3: oversample then probabilistically cull edge particles for natural falloff
    oversample = int(count * 1.25)
    theta = rng.uniform(0.0, math.tau, oversample)
    base_x, base_y = heart_boundary(theta)
    r_factor = 0.02 + 0.92 * (rng.random(oversample) ** 1.4)
    xy_noise_scale = 0.025 + 0.035 * (1.0 - r_factor)
    x = base_x * r_factor + rng.normal(0.0, xy_noise_scale, oversample)
    y = base_y * r_factor + rng.normal(0.0, xy_noise_scale, oversample)
    z = _z_depth(r_factor, theta, oversample, rng)

    positions = np.column_stack([x, y, z])
    # F3: edge culling — when r_factor > 0.80, survival probability decays linearly (was 0.85)
    edge_mask = r_factor > 0.80
    survival = np.ones(oversample)
    if edge_mask.any():
        survival[edge_mask] = 1.0 - (r_factor[edge_mask] - 0.80) / 0.14
        survival[edge_mask] = np.clip(survival[edge_mask], 0.03, 1.0)  # min survival lowered from 0.08
    keep = rng.random(oversample) < survival
    # Trim to exact count, preferring kept particles
    kept_idx = np.where(keep)[0]
    if len(kept_idx) > count:
        kept_idx = rng.choice(kept_idx, count, replace=False)
    elif len(kept_idx) < count:
        # If undershot, top up from culled particles
        culled_idx = np.where(~keep)[0]
        needed = count - len(kept_idx)
        extras = rng.choice(culled_idx, needed, replace=False)
        kept_idx = np.concatenate([kept_idx, extras])
    kept_idx.sort()
    positions = positions[kept_idx]
    r_factor = r_factor[kept_idx]
    theta = theta[kept_idx]
    z = z[kept_idx]
    xy_noise_scale = xy_noise_scale[kept_idx]

    n = count
    # H1: 3-stop gradient: deep red #c2185b → bright red #ff2d55 → light gold #ffb866
    deep_red = np.array([0.76, 0.09, 0.36], dtype="f4")
    bright_red = np.array([1.00, 0.18, 0.33], dtype="f4")
    light_gold = np.array([1.00, 0.72, 0.40], dtype="f4")
    nr = np.clip(r_factor / 0.94, 0.0, 1.0)
    nr += rng.normal(0.0, 0.08, n)
    nr = np.clip(nr, 0.0, 1.0)
    # Centre zone (low r_factor): deep_red→bright_red→gold
    # Edge zone (high r_factor): stays in deep_red range
    colors = _gradient_3stop(1.0 - nr, deep_red, bright_red, light_gold, 0.40)
    # C1: size tied to r_factor — centre noticeably larger, edge smaller
    size_factor = 0.55 + 0.45 * (1.0 - np.clip(r_factor / 0.88, 0.0, 1.0))
    sizes = rng.uniform(1.5, 3.0, n) * size_factor  # max reduced from 3.5
    phase = rng.uniform(0.0, math.tau, n)
    # C2: edge particles sharper falloff — stronger gradient for fill layer
    base_alpha = rng.uniform(0.22, 0.42, n)  # raised
    alpha_factor = 0.55 + 0.45 * (1.0 - np.clip(r_factor / 0.88, 0.0, 1.0))  # strengthened edge decay
    alpha = base_alpha * alpha_factor
    speed = rng.uniform(0.7, 1.5, n)
    return _stack_particles(positions, colors, sizes, phase, 2.0, r_factor, alpha, speed)


def _sample_core_new(rng, count):
    theta = rng.uniform(0.0, math.tau, count)
    phi = rng.uniform(0.0, math.pi, count)
    r = rng.random(count) ** (1.0 / 3.0) * 0.35
    r_factor = r / 0.35
    # D2: reshape core from sphere to heart contour using local heart radius
    local_r = _heart_local_radius(theta)
    heart_scale = np.clip(local_r / 1.17, 0.50, 1.0)
    hx = r * np.sin(phi) * np.cos(theta) * 0.95
    hy = r * np.sin(phi) * np.sin(theta) * 1.08
    x = hx * heart_scale + rng.normal(0.0, 0.020, count)  # tightened from 0.025
    y = hy * heart_scale + rng.normal(0.0, 0.020, count)  # tightened from 0.025
    z = r * np.cos(phi) * 0.75 + rng.normal(0.0, 0.03, count)

    positions = np.column_stack([x, y, z])
    n = count
    # B1: core colour gradient — bright pink-white centre ➜ warm red edge
    br = np.array([1.00, 0.69, 0.75], dtype="f4")  # #ffb0bf bright pink
    dr = np.array([1.00, 0.30, 0.43], dtype="f4")  # #ff4d6d warm red
    nr = np.clip(r_factor, 0.0, 1.0)
    nr += rng.normal(0.0, 0.10, n)
    nr = np.clip(nr, 0.0, 1.0)
    colors = _lerp_colors(br, dr, nr)
    # C1: size tied to r_factor — subtler gradient for core (all near centre)
    size_factor = 0.82 + 0.18 * (1.0 - r_factor)
    sizes = rng.uniform(3.5, 6.5, n) * size_factor
    phase = rng.uniform(0.0, math.tau, n)
    alpha = rng.uniform(0.42, 0.68, n)  # raised
    speed = rng.uniform(1.0, 2.2, n)
    return _stack_particles(positions, colors, sizes, phase, 3.0, r_factor, alpha, speed)


def _sample_trail_orbit(rng, count):
    """E2: Track particles that sit on the heart contour shell (slightly outside)
    and drift along it slowly, creating a luminous flowing trail around the heart."""
    theta = rng.uniform(0.0, math.tau, count)
    base_x, base_y = heart_boundary(theta)
    # Scale to sit just outside the outline
    shell = 1.08 + rng.random(count) * 0.22
    # Minimal jitter — keep them tight to the contour
    jitter = rng.normal(0.0, 0.010, (count, 2))
    x = base_x * shell + jitter[:, 0]
    y = base_y * shell + jitter[:, 1]
    # Z offset: spread around the mid-plane
    z = rng.normal(0.0, _HEART_DEPTH * 0.22, count)

    positions = np.column_stack([x, y, z])
    n = count
    # Bright warm gold-white colours for contrast against the red heart
    gold = np.array([1.00, 0.85, 0.35], dtype="f4")
    white = np.array([1.00, 0.95, 0.70], dtype="f4")
    mix = rng.random(n)
    colors = _lerp_colors(gold, white, mix)
    sizes = rng.uniform(1.0, 3.0, n)
    phase = rng.uniform(0.0, math.tau, n)  # orbital phase offset
    alpha = rng.uniform(0.12, 0.28, n)
    speed = rng.uniform(0.25, 0.65, n)  # slower than regular orbit — gentle drift
    return _stack_particles(positions, colors, sizes, phase, 4.5, np.zeros(n), alpha, speed)


def _sample_orbit_new(rng, count):
    angle = rng.uniform(0.0, math.tau, count)
    path_choice = rng.random(count)

    s_val = angle
    denom = 1.0 + np.sin(s_val) ** 2
    lem_x = 0.95 * np.cos(s_val) / denom
    lem_y = 0.55 * np.sin(s_val) * np.cos(s_val) / denom
    lem_z = 0.45 * np.sin(s_val) / denom

    r_circ = rng.uniform(1.05, 1.85, count)
    circ_x = r_circ * np.cos(angle)
    circ_y = rng.normal(0.0, 0.35, count)
    circ_z = r_circ * 0.6 * np.sin(angle)

    mask_lem = path_choice < 0.35
    mask_circ = (path_choice >= 0.35) & (path_choice < 0.75)
    mask_scatter = path_choice >= 0.75

    x = np.zeros(count)
    y = np.zeros(count)
    z = np.zeros(count)
    x[mask_lem] = lem_x[mask_lem]
    y[mask_lem] = lem_y[mask_lem]
    z[mask_lem] = lem_z[mask_lem]
    x[mask_circ] = circ_x[mask_circ]
    y[mask_circ] = circ_y[mask_circ]
    z[mask_circ] = circ_z[mask_circ]
    if mask_scatter.any():
        sc = mask_scatter.sum()
        sc_r = rng.uniform(1.1, 2.0, sc)
        sc_a = rng.uniform(0.0, math.tau, sc)
        sc_b = rng.uniform(0.0, math.tau, sc)
        x[mask_scatter] = sc_r * np.cos(sc_a) * np.cos(sc_b)
        y[mask_scatter] = sc_r * np.sin(sc_b) * 0.6
        z[mask_scatter] = sc_r * np.sin(sc_a) * np.cos(sc_b) * 0.5

    positions = np.column_stack([x, y, z])
    n = count
    distance = np.ones(n, dtype="f4") * 0.50
    radial = np.full(n, 0.15, dtype="f4")
    mix = np.clip(rng.normal(0.5, 0.22, n), 0.0, 1.0)
    colors = _lerp_colors(
        np.array([1.0, 0.60, 0.15], dtype="f4"),
        np.array([1.0, 0.85, 0.30], dtype="f4"),
        mix,
    )
    sizes = rng.uniform(1.5, 4.0, n)
    phase = rng.uniform(0.0, math.tau, n)
    alpha = rng.uniform(0.10, 0.20, n)
    speed = rng.uniform(0.8, 1.9, n)
    return _stack_particles(positions, colors, sizes, phase, 4.0, radial, alpha, speed)


def _sample_stars(rng, count):
    phi = rng.uniform(0.0, math.pi, count)
    theta = rng.uniform(0.0, math.tau, count)
    radius = rng.uniform(10.0, 20.0, count)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    # H2: subtle colour variation — bluish-white to warm-white
    cw = np.full((count, 3), 0.95, dtype="f4")
    cw[:, 0] = rng.uniform(0.80, 1.00, count)
    cw[:, 1] = rng.uniform(0.88, 1.00, count)
    cw[:, 2] = rng.uniform(0.92, 1.00, count)
    phase = rng.uniform(0.0, math.tau, count)
    speed = rng.uniform(0.03, 0.25, count)  # wider speed range for varied twinkle
    return _stack_particles(np.column_stack([x, y, z]), cw,
                            rng.uniform(1.5, 3.5, count),
                            phase, 5.0, np.zeros(count),
                            np.full(count, 0.7, dtype="f4"),
                            speed)


def _sample_sparks(rng, count):
    # H3: ~25% sparks concentrated near bottom tip for golden sparkle highlight
    n_uniform = int(count * 0.75)
    n_tip = count - n_uniform
    theta_u = rng.uniform(0.0, math.tau, n_uniform)
    tip_raw = rng.beta(4.0, 4.0, n_tip)
    theta_tip = 0.80 * math.pi + 0.40 * math.pi * tip_raw
    theta = np.concatenate([theta_u, theta_tip])
    rng.shuffle(theta)
    base_x, base_y = heart_boundary(theta)
    nx, ny = _heart_outward_normal(theta)
    shell = 0.96 + 0.06 * rng.random(count)
    jitter = rng.normal(0.0, 0.014, (count, 2))
    x = base_x * shell + jitter[:, 0]
    y = base_y * shell + jitter[:, 1]
    z = rng.normal(0.0, _HEART_DEPTH * 0.12, count)

    n = count
    dx = nx + rng.normal(0.0, 0.40, n)
    dy = ny + rng.normal(0.0, 0.40, n)
    dz = rng.normal(0.0, 0.40, n)
    dn = np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-8)
    dx /= dn; dy /= dn; dz /= dn

    sd = np.zeros(n, dtype=[("pos", "f4", 3), ("dir", "f4", 3), ("speed", "f4"),
                             ("p_off", "f4"), ("sz", "f4"), ("al", "f4")])
    sd["pos"] = np.column_stack([x, y, z]).astype("f4")
    sd["dir"] = np.column_stack([dx, dy, dz]).astype("f4")
    sd["speed"] = rng.uniform(0.3, 1.0, n).astype("f4")
    sd["p_off"] = rng.uniform(0.0, 0.15, n).astype("f4")
    sd["sz"] = rng.uniform(1.5, 4.0, n).astype("f4")
    sd["al"] = rng.uniform(0.5, 0.9, n).astype("f4")
    return sd


def _sample_rings(rng, count):
    n_rings = 6; per = count // n_rings
    rows = []
    for i in range(n_rings):
        br = 0.7 + i * 0.35
        nthis = per if i < n_rings - 1 else count - len(rows)
        ang = rng.uniform(0.0, math.tau, nthis)
        yo = rng.uniform(-0.15, 0.15, nthis)
        sp = rng.uniform(0.6, 1.0, nthis)
        for j in range(nthis):
            rows.append((math.cos(ang[j]) * br, yo[j], math.sin(ang[j]) * br, br, sp[j], 0.05))
    return np.array(rows, dtype="f4")


def build_particle_cloud(seed, total_count):
    rng = np.random.default_rng(seed)
    ratios = {
        "outline": 0.20,
        "loose_outline": 0.03,
        "shell": 0.22,
        "fill": 0.36,
        "core": 0.13,
        "orbit": 0.04,
        "trail_orbit": 0.02,
    }
    # Compute raw counts, round each; remainder goes to fill
    raw_counts = {k: int(total_count * v) for k, v in ratios.items()}
    raw_counts["fill"] = total_count - sum(v for k, v in raw_counts.items() if k != "fill")
    counts = raw_counts

    print(f"Generating {total_count} particles: "
          f"outline={counts['outline']}, loose_outline={counts['loose_outline']}, "
          f"shell={counts['shell']}, fill={counts['fill']}, "
          f"core={counts['core']}, orbit={counts['orbit']}, "
          f"trail_orbit={counts['trail_orbit']}")

    t0 = time.perf_counter()
    layers = [
        _sample_outline_new(rng, counts["outline"]),
        _sample_loose_outline(rng, counts["loose_outline"]),
        _sample_shell_new(rng, counts["shell"]),
        _sample_fill_new(rng, counts["fill"]),
        _sample_core_new(rng, counts["core"]),
        _sample_orbit_new(rng, counts["orbit"]),
        _sample_trail_orbit(rng, counts["trail_orbit"]),
    ]
    particles = np.vstack(layers).astype("f4")
    rng.shuffle(particles, axis=0)
    t1 = time.perf_counter()
    print(f"Particle generation: {t1 - t0:.2f}s, VBO size: {particles.nbytes / 1024 / 1024:.1f} MB")
    return particles


# ═══════════════════════════════════════════════════════════════
# Section 4: GLSL Shaders
# ═══════════════════════════════════════════════════════════════

QUAD_VS = """#version 330
const vec2 POS[6] = vec2[6](
    vec2(-1.0, -1.0), vec2( 1.0, -1.0), vec2(-1.0,  1.0),
    vec2(-1.0,  1.0), vec2( 1.0, -1.0), vec2( 1.0,  1.0)
);
out vec2 v_uv;
void main() {
    vec2 p = POS[gl_VertexID];
    gl_Position = vec4(p, 0.0, 1.0);
    v_uv = p * 0.5 + 0.5;
}
"""

PARTICLE_VS = """#version 330

uniform mat4 u_proj;
uniform mat4 u_view;
uniform float u_time;
uniform float u_beat;
uniform float u_point_scale;
uniform float u_shockwave;
// uniform float u_rotation;  // disabled — rotation off
uniform vec3 u_theme_tint[5];
uniform vec3 u_light_dir;    // G1: directional light for volume shading

in vec3 in_position;
in vec3 in_color;
in float in_size;
in float in_phase;
in float in_kind;
in float in_radial;
in float in_alpha;
in float in_speed;

out vec3 v_color;
out float v_alpha;
out float v_brightness;
out float v_kind;
out float v_core_mix;
out float v_seed;

float saturate(float v) { return clamp(v, 0.0, 1.0); }

vec3 rot_y(vec3 v, float a) {
    float c = cos(a), s = sin(a);
    return vec3(c*v.x + s*v.z, v.y, -s*v.x + c*v.z);
}
vec3 rot_x(vec3 v, float a) {
    float c = cos(a), s = sin(a);
    return vec3(v.x, c*v.y - s*v.z, s*v.y + c*v.z);
}

void main() {
    vec3 pos = in_position;

    // Body: Y-axis self-rotation + breathing wave (excludes orbit)
    if (in_kind < 3.5) {
        // pos = rot_y(pos, u_rotation);  // rotation disabled
        // E1: multi-axis breathing wave for organic feel
        pos.y += 0.035 * sin(u_time * 1.20 + in_phase * 0.45);
        pos.x += 0.018 * sin(u_time * 1.50 + in_phase * 0.35 + 0.8);
        pos.z += 0.015 * cos(u_time * 0.90 + in_phase * 0.50);
        // Original gentle vertical float (retained, layered on top)
        pos.y += 0.020 * sin(u_time * 0.65 + in_phase * 0.30);
    }

    float edge_lock = smoothstep(0.78, 1.0, in_radial);
    float core_mix = 1.0 - smoothstep(0.25, 0.95, in_radial);
    float pulse = u_beat;
    float shock = u_shockwave;
    float shimmer = 0.5 + 0.5 * sin(u_time * (1.2 + in_speed * 0.35) + in_phase);

    if (in_kind < 0.5) {
        // Outline: pulse + tiny z shimmer
        pos.xy *= 1.0 + pulse * 0.032;
        pos.z += 0.008 * sin(u_time * 0.75 + in_phase * 0.7);
    } else if (in_kind < 1.5) {
        // Shell: pure uniform pulse
        pos.xy *= 1.0 + pulse * 0.038;
        pos.z  *= 1.0 + pulse * 0.045;
    } else if (in_kind < 2.5) {
        // Fill: uniform pulse (stronger at centre)
        pos.xy *= 1.0 + pulse * (0.045 + 0.025 * core_mix);
        pos.z  *= 1.0 + pulse * 0.055;
        pos    *= 1.0 + shock * 0.005 * (1.0 - abs(in_radial - 0.5) * 2.0);
    } else if (in_kind < 3.5) {
        // Core: strongest uniform pulse
        pos.xy *= 1.0 + pulse * (0.055 + 0.035 * core_mix);
        pos.z  *= 1.0 + pulse * 0.065;
        pos    *= 1.0 + shock * 0.008 * core_mix;
    } else if (in_kind < 4.5) {
        // Orbit: gentle orbital drift
        float orbit = u_time * (0.85 + in_speed * 0.40) + in_phase;
        float c = cos(orbit), s = sin(orbit);
        float denom = 1.0 + s * s;
        float blend = 0.5 + 0.5 * sin(in_phase * 2.7);
        vec3 fig8 = vec3(
            in_position.x + 0.14 * c / denom,
            in_position.y + 0.14 * s * c / denom * 0.7,
            in_position.z + 0.10 * s / denom
        );
        vec3 circ = vec3(
            in_position.x + 0.05 * cos(orbit),
            in_position.y + 0.03 * sin(orbit * 1.2),
            in_position.z + 0.06 * sin(orbit)
        );
        pos = mix(circ, fig8, blend);
        pos *= 1.0 + pulse * 0.035;
        // Micro-tremor for natural feel
        pos.x += 0.010 * sin(u_time * 3.5 + in_phase * 5.0);
        pos.y += 0.010 * cos(u_time * 4.0 + in_phase * 4.5);
        pos.z += 0.008 * sin(u_time * 3.0 + in_phase * 5.5);
    } else {
        // E2: Trail-orbit — slow drift along heart contour shell
        float drift = u_time * (0.45 + in_speed * 0.30) + in_phase;
        // Approximate heart-contour-following motion via elliptical oscillation
        float cx = cos(drift);
        float sx = sin(drift);
        pos.x = in_position.x + 0.06 * cx;
        pos.y = in_position.y + 0.05 * sx * cx * 0.6;
        pos.z = in_position.z + 0.04 * sx;
        pos *= 1.0 + pulse * 0.028;
        // Subtle shimmer
        pos.x += 0.006 * sin(u_time * 5.0 + in_phase * 7.0);
        pos.y += 0.006 * cos(u_time * 4.5 + in_phase * 6.0);
    }

    // Camera-space transform
    vec4 view_pos = u_view * vec4(pos, 1.0);
    gl_Position = u_proj * view_pos;

    float depth = -view_pos.z;
    float frontness = saturate(pos.z * 1.15 + 0.5);
    float depth_scale = 1.55 / max(2.0, depth);
    float pulse_size = 1.0 + pulse * 0.06;
    float front_size = mix(0.84, 1.22, frontness);
    gl_PointSize = clamp(in_size * u_point_scale * depth_scale * pulse_size * front_size, 1.2, 35.0);

    // Lighting
    float depth_light = saturate(1.34 - 0.12 * abs(depth));  // brighter
    float sil_light = mix(0.90, 0.96, edge_lock);
    float shim_light = mix(0.95, 1.04, shimmer);
    vec3 depth_tint = mix(vec3(0.65, 0.72, 0.98), vec3(1.12, 1.00, 0.90), frontness);
    if (in_kind > 3.5) sil_light *= 1.03;
    // G1: directional light — top-left illumination creates volume
    float dir_light = dot(normalize(in_position), u_light_dir) * 0.5 + 0.5;
    dir_light = mix(0.64, 1.0, dir_light);  // brighter floor
    v_color = in_color * depth_tint * depth_light * sil_light * shim_light * dir_light
              * u_theme_tint[int(in_kind)];

    // Gentle centre glow (no flash)
    v_brightness = 1.06 + core_mix * (0.62 + u_beat * 0.42);  // brighter

    // Depth fog (mild, to keep particles visible)
    float fog = exp(-depth * 0.08);
    v_alpha = in_alpha * fog * mix(0.65, 1.25, frontness) * mix(0.90, 1.12, shimmer);
    v_kind = in_kind;
    v_core_mix = core_mix;
    v_seed = in_phase;
}
"""

PARTICLE_FS = """#version 330

in vec3 v_color;
in float v_alpha;
in float v_brightness;
in float v_kind;
in float v_core_mix;
in float v_seed;

out vec4 fragColor;

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) discard;

    float radius = sqrt(r2);
    float angle = atan(uv.y, uv.x);
    // G3: enhanced wobble — extra high-frequency layer for emboss-like micro-texture
    float wobble = 0.86
        + 0.06 * sin(angle * 5.0 + v_seed * 1.9)
        + 0.03 * sin(angle * 9.0 - v_seed * 1.3)
        + 0.02 * sin(angle * 13.0 + v_seed * 2.7);
    float particle_mask = 1.0 - smoothstep(wobble - 0.11, wobble, radius);  // tighter soft edge
    float dense = exp(-r2 * 7.0);
    float soft = exp(-r2 * 3.0);
    float halo = exp(-r2 * 2.0);

    float alpha_shape = particle_mask * mix(0.62, 1.0, dense);
    float glow = mix(0.95, 1.06, halo);
    // B2: radial falloff via pow(1 - radius, exp) — centre bright, edge soft
    float soft_falloff = 1.0;

    if (v_kind < 1.5) {
        alpha_shape *= mix(0.84, 1.0, dense);
        glow = mix(0.96, 1.03, halo);
        soft_falloff = pow(1.0 - radius, 1.3);  // outline: sharper edge
    } else if (v_kind < 3.5) {
        alpha_shape *= mix(0.86, 1.0, soft);
        glow = mix(0.95, 1.08, halo);
        soft_falloff = pow(1.0 - radius, 1.8);  // fill/core: softer edge
    } else if (v_kind < 4.5) {
        alpha_shape *= mix(0.82, 1.0, soft);
        glow = mix(1.00, 1.18, halo);  // reduced from 1.10~1.30
        soft_falloff = pow(1.0 - radius, 1.3);  // orbit: sharper
    } else {
        // E2: trail-orbit — brighter, more ethereal glow
        alpha_shape *= mix(0.76, 1.0, soft);
        glow = mix(1.05, 1.22, halo);  // reduced from 1.20~1.45
        soft_falloff = pow(1.0 - radius, 1.1);  // trail: brightest at centre
    }
    alpha_shape *= soft_falloff;
    glow *= mix(1.0, 1.06, v_core_mix);
    // Per-particle glow variation for natural sparkle
    float glow_var = 0.85 + 0.30 * sin(v_seed * 7.3);
    glow *= glow_var;

    // G2: inner warm glow + outer deep-red aura for layered particle feel
    float inner_glow = exp(-r2 * 12.0);
    float outer_glow = exp(-r2 * 1.5);
    vec3 inner_color = vec3(1.0, 0.70, 0.30);  // warm orange core
    vec3 outer_color = vec3(0.70, 0.00, 0.20); // deep red aura

    // HDR color: brightness can exceed 1.0 for bloom extraction
    vec3 hdr = v_color * glow * v_brightness
             + inner_color * inner_glow * 0.22  // raised from 0.15
             + outer_color * outer_glow * 0.045;  // raised from 0.03
    fragColor = vec4(hdr, v_alpha * alpha_shape);
}
"""

BRIGHT_PASS_FS = """#version 330
uniform sampler2D u_scene_tex;
uniform float u_threshold;
uniform float u_intensity;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec3 color = texture(u_scene_tex, v_uv).rgb;
    float lum = dot(color, vec3(0.2126, 0.7152, 0.0722));
    float contrib = smoothstep(u_threshold - 0.15, u_threshold, lum);
    fragColor = vec4(color * contrib * u_intensity, 1.0);
}
"""

BLUR_H_FS = """#version 330
uniform sampler2D u_input_tex;
uniform vec2 u_texel_size;
in vec2 v_uv;
out vec4 fragColor;

const float W[9] = float[9](
    0.0136, 0.0476, 0.1172, 0.2028, 0.2378, 0.2028, 0.1172, 0.0476, 0.0136
);

void main() {
    vec3 result = vec3(0.0);
    for (int i = 0; i < 9; i++) {
        vec2 uv = v_uv + vec2(u_texel_size.x * float(i - 4), 0.0);
        result += texture(u_input_tex, uv).rgb * W[i];
    }
    fragColor = vec4(result, 1.0);
}
"""

BLUR_V_FS = """#version 330
uniform sampler2D u_input_tex;
uniform vec2 u_texel_size;
in vec2 v_uv;
out vec4 fragColor;

const float W[9] = float[9](
    0.0136, 0.0476, 0.1172, 0.2028, 0.2378, 0.2028, 0.1172, 0.0476, 0.0136
);

void main() {
    vec3 result = vec3(0.0);
    for (int i = 0; i < 9; i++) {
        vec2 uv = v_uv + vec2(0.0, u_texel_size.y * float(i - 4));
        result += texture(u_input_tex, uv).rgb * W[i];
    }
    fragColor = vec4(result, 1.0);
}
"""

COMPOSITE_FS = """#version 330
uniform sampler2D u_scene_tex;
uniform sampler2D u_bloom_tex;
uniform sampler2D u_trail_tex;
uniform float u_bloom_strength;
uniform float u_trail_strength;
uniform float u_time;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    vec3 scene = texture(u_scene_tex, v_uv).rgb;
    vec3 bloom = texture(u_bloom_tex, v_uv).rgb;
    vec3 trail = texture(u_trail_tex, v_uv).rgb;

    // HDR composition
    vec3 hdr = scene + bloom * u_bloom_strength + trail * u_trail_strength;

    // Reinhard tonemap
    vec3 ldr = hdr / (hdr + vec3(1.0));

    // Subtle vignette
    float vignette = 1.0 - 0.3 * length(v_uv * 2.0 - 1.0);
    vignette = smoothstep(0.0, 1.0, vignette);
    ldr *= vignette;

    // H2: subtle radial background glow (deep purple) — reduced intensity
    float dist = length(v_uv * 2.0 - 1.0);
    vec3 bg_glow = vec3(0.022, 0.006, 0.040) * (1.0 - dist * 0.6);  // raised from (0.015,0.004,0.03)
    ldr += bg_glow;

    // H2: film grain for 'night sky' texture
    float grain = fract(sin(dot(v_uv, vec2(12.9898, 78.233))) * 43758.5453);
    grain = (grain - 0.5) * 0.006;  // reduced from 0.012
    ldr += grain;

    // Subtle filmic contrast
    ldr = smoothstep(0.0, 1.0, ldr);

    fragColor = vec4(ldr, 1.0);
}
"""

TRAIL_DECAY_FS = """#version 330
uniform sampler2D u_input_tex;
uniform float u_decay;
in vec2 v_uv;
out vec4 fragColor;

void main() {
    fragColor = texture(u_input_tex, v_uv) * u_decay;
}
"""
STAR_VS = """#version 330
uniform mat4 u_proj;
uniform mat4 u_view;
uniform float u_time;

in vec3 in_position;
in vec3 in_color;
in float in_size;
in float in_phase;
in float in_kind;
in float in_radial;
in float in_alpha;
in float in_speed;

out vec3 v_color;
out float v_alpha;

void main() {
    vec4 view_pos = u_view * vec4(in_position, 1.0);
    gl_Position = u_proj * view_pos;
    gl_PointSize = in_size * 2.5 / max(1.0, -view_pos.z);

    float twinkle = 0.6 + 0.4 * sin(u_time * in_speed + in_phase);
    v_color = in_color * (0.95 + 0.05 * in_kind * 0.0);  // use in_kind (prevent opt-out)
    float rad_factor = 1.0 - 0.02 * in_radial;  // use in_radial (prevent opt-out)
    v_alpha = in_alpha * twinkle * rad_factor;
}
"""

STAR_FS = """#version 330
in vec3 v_color;
in float v_alpha;
out vec4 fragColor;

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float d = length(uv);
    if (d > 1.0) discard;
    float soft = exp(-d * d * 6.0);
    fragColor = vec4(v_color, v_alpha * soft);
}
"""

SPARK_VS = """#version 330
uniform mat4 u_proj;
uniform mat4 u_view;
uniform float u_time;

in vec3 in_origin;
in vec3 in_direction;
in float in_speed;
in float in_phase_off;
in float in_particle_size;
in float in_alpha_val;

out vec3 v_color;
out float v_alpha;

void main() {
    // Heartbeat spark burst
    float period = 60.0 / 72.0;
    float phase = mod(u_time / period, 1.0);
    float activation = 0.10 + in_phase_off;
    float elapsed = phase - activation;
    float is_active = step(0.0, elapsed);

    float age = clamp(elapsed / 0.30, 0.0, 1.0);
    float life = 1.0 - age;

    vec3 pos = in_origin + in_direction * in_speed * age * 2.5;

    vec4 view_pos = u_view * vec4(pos, 1.0);
    gl_Position = u_proj * view_pos;

    float spark_alpha = is_active * smoothstep(0.0, 0.15, life) * (1.0 - smoothstep(0.5, 1.0, age));
    gl_PointSize = max(0.5, in_particle_size * (1.0 - age * 0.7)) * 2.5 / max(1.0, -view_pos.z);

    // H3: richer gold sparkle — warmer and more luminous
    vec3 core_gold = vec3(1.0, 0.88, 0.50);
    vec3 tip_white = vec3(1.0, 0.96, 0.82);
    v_color = mix(core_gold, tip_white, age);
    v_alpha = in_alpha_val * spark_alpha;
}
"""

SPARK_FS = """#version 330
in vec3 v_color;
in float v_alpha;
out vec4 fragColor;

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float d = length(uv);
    if (d > 1.0) discard;
    float soft = exp(-d * d * 4.0);
    float glow = exp(-d * d * 8.0);
    vec3 col = v_color * (1.0 + glow * 0.6);
    fragColor = vec4(col, v_alpha * soft);
}
"""

RING_VS = """#version 330
uniform mat4 u_proj;
uniform mat4 u_view;
uniform float u_time;

in vec3 in_position;
in float in_base_radius;
in float in_expand_speed;
in float in_ring_phase;

out vec3 v_color;
out float v_alpha;

void main() {
    float period = 60.0 / 72.0;
    float phase = mod(u_time / period, 1.0);

    float activation = 0.10 + in_ring_phase * 0.05;
    float ring_age = (phase - activation) / 0.50;
    ring_age = clamp(ring_age, 0.0, 1.0);
    float ring_visible = step(activation, phase) * step(0.0, activation + 0.50 - phase);

    // Expand outward using expand_speed and base_radius
    float expand = 1.0 + ring_age * in_expand_speed;
    vec3 pos = in_position * expand;
    float radius_factor = in_base_radius / 2.1;

    vec4 view_pos = u_view * vec4(pos, 1.0);
    gl_Position = u_proj * view_pos;

    float ring_alpha = (1.0 - ring_age) * 0.35 * ring_visible;
    gl_PointSize = max(0.5, (3.0 - ring_age * 2.0)) * 2.0 / max(1.0, -view_pos.z);

    // Soft blue-white ring color (theme will be applied via tint in app)
    vec3 inner_color = vec3(0.8, 0.9, 1.0);
    vec3 outer_color = vec3(0.5, 0.7, 0.9);
    v_color = mix(inner_color, outer_color, radius_factor);
    v_alpha = ring_alpha;
}
"""

RING_FS = """#version 330
in vec3 v_color;
in float v_alpha;
out vec4 fragColor;

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float d = length(uv);
    if (d > 1.0) discard;
    float soft = exp(-d * d * 3.0);
    float glow = exp(-d * d * 6.0);
    vec3 col = v_color * (1.0 + glow * 0.8);
    fragColor = vec4(col, v_alpha * soft);
}
"""




# ═══════════════════════════════════════════════════════════════
# Section 5: Camera System
# ═══════════════════════════════════════════════════════════════

class CinematicCamera:
    """Camera supporting auto-cinematic orbit and manual user control."""

    AUTO = "auto"
    MANUAL = "manual"
    TRANSITION = "transition"

    def __init__(self) -> None:
        self._view = np.eye(4, dtype="f4")
        self._proj = np.eye(4, dtype="f4")
        self.mode = self.AUTO
        self._transition_time = 0.0
        self._transition_duration = 0.5

        # Manual control state
        self.manual_yaw = 0.0
        self.manual_pitch = 0.0
        self.manual_radius = 4.0
        self.manual_tx = 0.0
        self.manual_ty = 0.0
        self.manual_tz = 0.0

        # Auto values (captured during transition)
        self._auto_yaw = 0.0
        self._auto_pitch = 0.0
        self._auto_radius = 4.0
        self._auto_tx = 0.0
        self._auto_ty = 0.0
        self._auto_tz = 0.0

    def switch_mode(self, new_mode: str, elapsed: float) -> None:
        if new_mode == self.MANUAL and self.mode != self.MANUAL:
            self._transition_time = elapsed
            self.mode = self.TRANSITION
            self._transition_target_mode = self.MANUAL
        elif new_mode == self.AUTO and self.mode != self.AUTO:
            self._transition_time = elapsed
            self.mode = self.TRANSITION
            self._transition_target_mode = self.AUTO
        else:
            self.mode = new_mode

    def set_manual(self, yaw: float, pitch: float, radius: float,
                   tx: float = 0.0, ty: float = 0.0, tz: float = 0.0) -> None:
        self.manual_yaw = yaw
        self.manual_pitch = max(-1.3, min(1.3, pitch))
        self.manual_radius = max(1.5, min(15.0, radius))
        self.manual_tx = tx
        self.manual_ty = ty
        self.manual_tz = tz

    def reset(self, aspect: float) -> None:
        """Reset to default position."""
        self.set_manual(0.0, 0.0, 4.0)
        self._rebuild(aspect, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0)

    def _rebuild(self, aspect, yaw, pitch, radius, tx, ty, tz):
        eye = np.array([
            radius * math.cos(pitch) * math.sin(yaw) + tx,
            radius * math.sin(pitch) + ty,
            radius * math.cos(pitch) * math.cos(yaw) + tz,
        ], dtype="f4")
        target = np.array([tx, ty, tz], dtype="f4")
        up = np.array([0.0, 1.0, 0.0], dtype="f4")
        self._view = self._look_at(eye, target, up)
        self._proj = self._perspective(math.radians(38.0), aspect, 0.1, 20.0)  # tighter FOV

    def update(self, elapsed: float, aspect: float):
        # Static camera — no rotation for a stable, full view of the heart
        a_yaw = 0.0
        a_pitch = 0.0
        a_radius = 3.55 + 0.12 * math.sin(elapsed * 0.18 + 0.5) + 0.06 * math.sin(elapsed * 0.31 + 2.1)  # tighter baseline & swing
        a_tx = 0.0
        a_ty = 0.010 * math.sin(elapsed * 0.55)  # reduced from 0.025
        a_tz = 0.0

        if self.mode == self.AUTO:
            self._rebuild(aspect, a_yaw, a_pitch, a_radius, a_tx, a_ty, a_tz)
        elif self.mode == self.MANUAL:
            self._rebuild(aspect, self.manual_yaw, self.manual_pitch,
                          self.manual_radius, self.manual_tx, self.manual_ty,
                          self.manual_tz)
        elif self.mode == self.TRANSITION:
            t = min(1.0, (elapsed - self._transition_time) / self._transition_duration)
            t_smooth = t * t * (3.0 - 2.0 * t)  # smoothstep
            if self._transition_target_mode == self.MANUAL:
                # From auto to manual: auto values were set when transition started
                yaw = self._auto_yaw + (self.manual_yaw - self._auto_yaw) * t_smooth
                pitch = self._auto_pitch + (self.manual_pitch - self._auto_pitch) * t_smooth
                radius = self._auto_radius + (self.manual_radius - self._auto_radius) * t_smooth
                tx = self._auto_tx + (self.manual_tx - self._auto_tx) * t_smooth
                ty = self._auto_ty + (self.manual_ty - self._auto_ty) * t_smooth
                tz = self._auto_tz + (self.manual_tz - self._auto_tz) * t_smooth
            else:
                # From manual to auto
                yaw = self.manual_yaw + (a_yaw - self.manual_yaw) * t_smooth
                pitch = self.manual_pitch + (a_pitch - self.manual_pitch) * t_smooth
                radius = self.manual_radius + (a_radius - self.manual_radius) * t_smooth
                tx = self.manual_tx + (a_tx - self.manual_tx) * t_smooth
                ty = self.manual_ty + (a_ty - self.manual_ty) * t_smooth
                tz = self.manual_tz + (a_tz - self.manual_tz) * t_smooth
            self._rebuild(aspect, yaw, pitch, radius, tx, ty, tz)

            if t >= 1.0:
                self.mode = self._transition_target_mode
        else:
            # Fallback: treat as auto
            self.mode = self.AUTO
            self._rebuild(aspect, a_yaw, a_pitch, a_radius, a_tx, a_ty, a_tz)

    @property
    def view(self) -> np.ndarray:
        return self._view

    @property
    def proj(self) -> np.ndarray:
        return self._proj

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v if n == 0.0 else v / n

    @staticmethod
    def _perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
        f = 1.0 / math.tan(fov_y / 2.0)
        m = np.zeros((4, 4), dtype="f4")
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (far + near) / (near - far)
        m[2, 3] = (2.0 * far * near) / (near - far)
        m[3, 2] = -1.0
        return m

    @classmethod
    def _look_at(cls, eye, target, up):
        fwd = cls._normalize(target - eye)
        side = cls._normalize(np.cross(fwd, up))
        up_c = np.cross(side, fwd)
        m = np.eye(4, dtype="f4")
        m[0, :3] = side
        m[1, :3] = up_c
        m[2, :3] = -fwd
        m[:3, 3] = -m[:3, :3] @ eye
        return m


# ═══════════════════════════════════════════════════════════════
# Section 6: Heartbeat System
# ═══════════════════════════════════════════════════════════════

class HeartbeatSystem:
    def __init__(self, bpm: float = 72.0) -> None:
        self._period = 60.0 / bpm
        self.beat = 0.0
        self.shockwave = 0.0
        self.bloom_intensity = 0.0

    def update(self, elapsed: float) -> None:
        # Bimodal lub-dub pulse: two Gaussian peaks per cycle
        phase = (elapsed % self._period) / self._period
        p1 = math.exp(-((phase - 0.12) ** 2) / 0.0035)       # "lub" — strong systolic peak
        p2 = math.exp(-((phase - 0.30) ** 2) / 0.0055) * 0.5  # "dub" — weaker diastolic peak
        # E3: added micro-rebound after dub
        p3 = math.exp(-((phase - 0.37) ** 2) / 0.008) * 0.25
        self.beat = 0.28 + 0.72 * (p1 + p2 + p3 * 0.3)
        self.shockwave = p1 * 0.85 + p3  # stronger shockwave with dub rebound
        self.bloom_intensity = 0.30 + 0.18 * (p1 + p2 * 0.6 + p3 * 0.4)  # raised from 0.24+0.16


# ═══════════════════════════════════════════════════════════════
# Section 7: Bloom Pipeline
# ═══════════════════════════════════════════════════════════════

class BloomPipeline:
    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        self._ctx = ctx
        self._width = width
        self._height = height
        self._hw = max(1, width // 2)
        self._hh = max(1, height // 2)

        self._setup_textures()
        self._setup_shaders()
        self._setup_quads()

    def _setup_textures(self) -> None:
        ctx = self._ctx
        w, h = self._width, self._height
        hw, hh = self._hw, self._hh

        self.tex_scene = ctx.texture((w, h), 4, dtype="f2")
        self.depth_rbo = ctx.depth_renderbuffer((w, h))
        self.fbo_scene = ctx.framebuffer(
            color_attachments=[self.tex_scene],
            depth_attachment=self.depth_rbo,
        )

        self.tex_bright_a = ctx.texture((hw, hh), 4, dtype="f2")
        self.fbo_bright_a = ctx.framebuffer(color_attachments=[self.tex_bright_a])

        self.tex_bright_b = ctx.texture((hw, hh), 4, dtype="f2")
        self.fbo_bright_b = ctx.framebuffer(color_attachments=[self.tex_bright_b])

    def _setup_shaders(self) -> None:
        self.prog_bright = self._ctx.program(
            vertex_shader=QUAD_VS, fragment_shader=BRIGHT_PASS_FS
        )
        self.prog_blur_h = self._ctx.program(
            vertex_shader=QUAD_VS, fragment_shader=BLUR_H_FS
        )
        self.prog_blur_v = self._ctx.program(
            vertex_shader=QUAD_VS, fragment_shader=BLUR_V_FS
        )
        self.prog_composite = self._ctx.program(
            vertex_shader=QUAD_VS, fragment_shader=COMPOSITE_FS
        )

    def _setup_quads(self) -> None:
        self._quad_bright = self._ctx.vertex_array(self.prog_bright, [])
        self._quad_blur_h = self._ctx.vertex_array(self.prog_blur_h, [])
        self._quad_blur_v = self._ctx.vertex_array(self.prog_blur_v, [])
        self._quad_composite = self._ctx.vertex_array(self.prog_composite, [])

    def begin_scene_pass(self) -> None:
        self.fbo_scene.use()
        self.fbo_scene.clear(*BACKGROUND, depth=1.0)

    def execute_post_passes(self, trail_tex, trail_strength, bloom_intensity) -> None:
        ctx = self._ctx
        hw, hh = self._hw, self._hh

        # Bright pass
        self.fbo_bright_a.use()
        ctx.viewport = (0, 0, hw, hh)
        self.fbo_bright_a.clear(0.0, 0.0, 0.0, 1.0)
        self.tex_scene.use(location=0)
        self.prog_bright["u_threshold"].value = 0.62  # lowered from 0.68
        self.prog_bright["u_intensity"].value = bloom_intensity
        self._quad_bright.render(moderngl.TRIANGLES, vertices=6)

        # Horizontal blur
        self.fbo_bright_b.use()
        self.fbo_bright_b.clear(0.0, 0.0, 0.0, 1.0)
        self.tex_bright_a.use(location=0)
        self.prog_blur_h["u_texel_size"].value = (1.0 / hw, 0.0)
        self._quad_blur_h.render(moderngl.TRIANGLES, vertices=6)

        # Vertical blur → back to bright_a
        self.fbo_bright_a.use()
        self.fbo_bright_a.clear(0.0, 0.0, 0.0, 1.0)
        self.tex_bright_b.use(location=0)
        self.prog_blur_v["u_texel_size"].value = (0.0, 1.0 / hh)
        self._quad_blur_v.render(moderngl.TRIANGLES, vertices=6)

        # Composite to screen
        ctx.screen.use()
        ctx.viewport = (0, 0, self._width, self._height)
        ctx.clear(*BACKGROUND, depth=1.0)
        self.tex_scene.use(location=0)
        self.tex_bright_a.use(location=1)
        if trail_tex is not None:
            trail_tex.use(location=2)
        else:
            self.tex_bright_a.use(location=2)
        self.prog_composite["u_bloom_strength"].value = bloom_intensity
        self.prog_composite["u_trail_strength"].value = trail_strength
        self._quad_composite.render(moderngl.TRIANGLES, vertices=6)

    def resize(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self._hw = max(1, width // 2)
        self._hh = max(1, height // 2)
        self.tex_scene.release()
        self.depth_rbo.release()
        self.fbo_scene.release()
        self.tex_bright_a.release()
        self.fbo_bright_a.release()
        self.tex_bright_b.release()
        self.fbo_bright_b.release()
        self._setup_textures()

    def release(self) -> None:
        for attr in ["tex_scene", "depth_rbo", "fbo_scene",
                     "tex_bright_a", "fbo_bright_a",
                     "tex_bright_b", "fbo_bright_b"]:
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()
        for attr in ["prog_bright", "prog_blur_h", "prog_blur_v", "prog_composite"]:
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()


# ═══════════════════════════════════════════════════════════════
# Section 8: Trail System (ping-pong accumulation)
# ═══════════════════════════════════════════════════════════════

class TrailSystem:
    DECAY = 0.92  # I1: longer trail persistence for motion-blur feel

    def __init__(self, ctx: moderngl.Context, width: int, height: int) -> None:
        self._ctx = ctx
        self._width = width
        self._height = height
        self._setup()

    def _setup(self) -> None:
        ctx = self._ctx
        w, h = self._width, self._height
        self.tex_a = ctx.texture((w, h), 4, dtype="f2")
        self.fbo_a = ctx.framebuffer(color_attachments=[self.tex_a])
        self.fbo_a.clear(0.0, 0.0, 0.0, 1.0)

        self.tex_b = ctx.texture((w, h), 4, dtype="f2")
        self.fbo_b = ctx.framebuffer(color_attachments=[self.tex_b])
        self.fbo_b.clear(0.0, 0.0, 0.0, 1.0)

        self.prog_decay = ctx.program(
            vertex_shader=QUAD_VS, fragment_shader=TRAIL_DECAY_FS
        )
        self.quad_decay = ctx.vertex_array(self.prog_decay, [])
        self._read_tex = self.tex_a
        self._write_fbo = self.fbo_b
        self._first_frame = True

    def accumulate(self, scene_tex) -> None:
        """Ping-pong: read from _read_tex, decay, add scene, write to _write_fbo."""
        if self._first_frame:
            # First frame: just copy scene to trail
            self._first_frame = False
            self._write_fbo.use()
            self._ctx.viewport = (0, 0, self._width, self._height)
            self._ctx.clear(0.0, 0.0, 0.0, 1.0)
            self._read_tex = self.tex_b
            self._write_fbo = self.fbo_a
            return

        self._write_fbo.use()
        self._ctx.viewport = (0, 0, self._width, self._height)
        self._ctx.clear(0.0, 0.0, 0.0, 1.0)

        # Decay previous frame
        # Render to _write_fbo: sample _read_tex * DECAY
        self._read_tex.use(location=0)
        self.prog_decay["u_input_tex"] = 0
        self.prog_decay["u_decay"].value = self.DECAY
        self.quad_decay.render(moderngl.TRIANGLES, vertices=6)

        # Add current scene with additive blend
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        # Render scene tex into the same FBO with a simple copy
        scene_tex.use(location=0)
        self.prog_decay["u_decay"].value = 0.12
        self.quad_decay.render(moderngl.TRIANGLES, vertices=6)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Swap
        self._read_tex, self._write_fbo = (
            self.tex_b if self._read_tex is self.tex_a else self.tex_a,
            self.fbo_a if self._write_fbo is self.fbo_b else self.fbo_b,
        )

    @property
    def trail_texture(self):
        return self._read_tex

    def resize(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self.tex_a.release()
        self.fbo_a.release()
        self.tex_b.release()
        self.fbo_b.release()
        self._first_frame = True
        self._setup()

    def release(self) -> None:
        for attr in ["tex_a", "fbo_a", "tex_b", "fbo_b", "prog_decay"]:
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()


# ═══════════════════════════════════════════════════════════════
# Section 9: Main Application
# ═══════════════════════════════════════════════════════════════

class ParticleHeartV2App:
    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._enable_bloom = args.bloom
        self._enable_trails = args.trails
        self._show_fps = args.fps
        self._self_test = args.self_test
        self._fb_size = (-1, -1)
        self._frame_count = 0
        self._fps_timer = 0.0
        self._fps_value = 0
        self._paused = False
        self._start_off = 0.0
        self._current_theme = args.theme
        self._target_theme = args.theme
        self._theme_blend = 0.0
        # Mouse state
        self._mouse_buttons = [False, False, False]
        self._mouse_x = 0.0
        self._mouse_y = 0.0
        self._mouse_dx = 0.0
        self._mouse_dy = 0.0
        self._scroll_offset = 0.0

        if not glfw.init():
            raise RuntimeError("GLFW init failed")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.DEPTH_BITS, 24)
        if sys.platform == "darwin":
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        if self._self_test:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        self._window = glfw.create_window(args.width, args.height, WINDOW_TITLE, None, None)
        if not self._window:
            glfw.terminate()
            raise RuntimeError("Failed to create window")
        glfw.make_context_current(self._window)
        glfw.swap_interval(1 if not self._self_test else 0)

        self._ctx = moderngl.create_context()
        self._ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE | moderngl.DEPTH_TEST)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self._setup_particles()
        self._setup_bloom()
        self._setup_trail()
        self._camera = CinematicCamera()
        self._heartbeat = HeartbeatSystem()
        self._update_fb_state()
        self._setup_stars()
        self._setup_sparks()
        self._setup_rings()

    def _setup_particles(self) -> None:
        particles = build_particle_cloud(self._args.seed, self._args.particles)
        self._particle_count = particles.shape[0]

        self._prog_particles = self._ctx.program(
            vertex_shader=PARTICLE_VS, fragment_shader=PARTICLE_FS
        )
        self._u_proj = self._prog_particles["u_proj"]
        self._u_view = self._prog_particles["u_view"]
        self._u_time = self._prog_particles["u_time"]
        self._u_beat = self._prog_particles["u_beat"]
        self._u_point_scale = self._prog_particles["u_point_scale"]
        self._u_shockwave = self._prog_particles["u_shockwave"]
        # self._u_rotation = self._prog_particles["u_rotation"]  # rotation disabled
        self._u_theme_tint = self._prog_particles["u_theme_tint"]
        self._u_light_dir = self._prog_particles["u_light_dir"]   # G1

        vbo = self._ctx.buffer(particles.tobytes())
        self._particle_vao = self._ctx.vertex_array(
            self._prog_particles,
            [(vbo, "3f 3f 1f 1f 1f 1f 1f 1f",
              "in_position", "in_color", "in_size", "in_phase",
              "in_kind", "in_radial", "in_alpha", "in_speed")],
        )
        self._vbo = vbo

    def _setup_bloom(self) -> None:
        if self._enable_bloom:
            self._bloom = BloomPipeline(self._ctx, self._args.width, self._args.height)
        else:
            self._bloom = None

    def _setup_trail(self) -> None:
        if self._enable_trails:
            self._trail = TrailSystem(self._ctx, self._args.width, self._args.height)
        else:
            self._trail = None

    def _setup_stars(self) -> None:
        rng = np.random.default_rng(self._args.seed + 10)
        stars = _sample_stars(rng, _BASE_STARS)
        self._prog_stars = self._ctx.program(vertex_shader=STAR_VS, fragment_shader=STAR_FS)
        vbo = self._ctx.buffer(stars.tobytes())
        self._star_vao = self._ctx.vertex_array(
            self._prog_stars,
            [(vbo, "3f 3f 1f 1f 1f 1f 1f 1f",
              "in_position", "in_color", "in_size", "in_phase",
              "in_kind", "in_radial", "in_alpha", "in_speed")],
        )
        self._star_vbo = vbo
        self._star_u_proj = self._prog_stars["u_proj"]
        self._star_u_view = self._prog_stars["u_view"]
        self._star_u_time = self._prog_stars["u_time"]

    def _setup_sparks(self) -> None:
        rng = np.random.default_rng(self._args.seed + 20)
        sparks = _sample_sparks(rng, _BASE_SPARKS)
        self._prog_sparks = self._ctx.program(vertex_shader=SPARK_VS, fragment_shader=SPARK_FS)
        vbo = self._ctx.buffer(sparks.tobytes())
        self._spark_vao = self._ctx.vertex_array(
            self._prog_sparks,
            [(vbo, "3f 3f 1f 1f 1f 1f",
              "in_origin", "in_direction", "in_speed",
              "in_phase_off", "in_particle_size", "in_alpha_val")],
        )
        self._spark_vbo = vbo
        self._spark_u_proj = self._prog_sparks["u_proj"]
        self._spark_u_view = self._prog_sparks["u_view"]
        self._spark_u_time = self._prog_sparks["u_time"]

    def _setup_rings(self) -> None:
        rng = np.random.default_rng(self._args.seed + 30)
        rings = _sample_rings(rng, _BASE_RINGS)
        self._prog_rings = self._ctx.program(vertex_shader=RING_VS, fragment_shader=RING_FS)
        vbo = self._ctx.buffer(rings.tobytes())
        self._ring_vao = self._ctx.vertex_array(
            self._prog_rings,
            [(vbo, "3f 1f 1f 1f",
              "in_position", "in_base_radius", "in_expand_speed", "in_ring_phase")],
        )
        self._ring_vbo = vbo
        self._ring_u_proj = self._prog_rings["u_proj"]
        self._ring_u_view = self._prog_rings["u_view"]
        self._ring_u_time = self._prog_rings["u_time"]

    def _update_fb_state(self) -> None:
        w, h = glfw.get_framebuffer_size(self._window)
        w = max(w, 1)
        h = max(h, 1)
        if (w, h) == self._fb_size:
            return
        self._fb_size = (w, h)
        self._ctx.viewport = (0, 0, w, h)
        if self._bloom is not None:
            self._bloom.resize(w, h)
        if self._trail is not None:
            self._trail.resize(w, h)

    def render(self, elapsed: float) -> None:
        self._update_fb_state()
        w, h = self._fb_size
        aspect = w / h

        # Handle theme transition
        if self._target_theme != self._current_theme:
            self._theme_blend = min(1.0, self._theme_blend + 0.03)
            if self._theme_blend >= 1.0:
                self._current_theme = self._target_theme
                self._theme_blend = 0.0

        self._camera.update(elapsed, aspect)
        self._heartbeat.update(elapsed)

        view = self._camera.view
        proj = self._camera.proj
        beat = self._heartbeat.beat
        shockwave = self._heartbeat.shockwave
        bloom_intensity = self._heartbeat.bloom_intensity
        point_scale = min(w, h) / 40.0

        # Compute active theme tints
        src = THEMES[self._current_theme]
        dst = THEMES[self._target_theme]
        blend = self._theme_blend
        theme_tints = []
        for k in range(5):
            if blend > 0:
                t = tuple((1 - blend) * src[k][i] + blend * dst[k][i] for i in range(3))
            else:
                t = src[k]
            theme_tints.append(t)
        self._u_theme_tint.value = tuple(theme_tints)  # tuple of 5 (r,g,b) rows, not flat

        # Pass 1: Scene (with bloom target)
        if self._enable_bloom:
            self._bloom.begin_scene_pass()
        else:
            self._ctx.screen.use()
            self._ctx.viewport = (0, 0, w, h)
            self._ctx.clear(*BACKGROUND, depth=1.0)

        # Starfield background (no depth write, in scene pass)
        if hasattr(self, '_star_vao'):
            self._ctx.depth_mask = False
            self._star_u_proj.write(proj.T.tobytes())
            self._star_u_view.write(view.T.tobytes())
            self._star_u_time.value = elapsed
            self._star_vao.render(moderngl.POINTS)
            self._ctx.depth_mask = True

        self._u_proj.write(proj.T.tobytes())
        self._u_view.write(view.T.tobytes())
        self._u_time.value = elapsed
        self._u_beat.value = beat
        self._u_point_scale.value = point_scale
        self._u_shockwave.value = shockwave
        # self._u_rotation.value = elapsed * _ROTATION_SPEED  # rotation disabled
        # G1: directional light from top-left-front
        self._u_light_dir.value = (0.45, 0.55, 0.65)
        self._particle_vao.render(moderngl.POINTS)

        # Render sparks (additive blend)
        if hasattr(self, '_spark_vao'):
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            self._spark_u_proj.write(proj.T.tobytes())
            self._spark_u_view.write(view.T.tobytes())
            self._spark_u_time.value = elapsed
            self._spark_vao.render(moderngl.POINTS)
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Render rings (additive blend)
        if hasattr(self, '_ring_vao'):
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            self._ring_u_proj.write(proj.T.tobytes())
            self._ring_u_view.write(view.T.tobytes())
            self._ring_u_time.value = elapsed
            self._ring_vao.render(moderngl.POINTS)
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        # Trail accumulation
        if self._trail is not None and self._enable_bloom:
            self._trail.accumulate(self._bloom.tex_scene)

        # Post-process passes
        trail_tex = self._trail.trail_texture if self._trail is not None else None
        trail_strength = 0.16 if self._trail is not None else 0.0  # raised from 0.12
        if self._enable_bloom:
            self._bloom.execute_post_passes(trail_tex, trail_strength, bloom_intensity)

    def run(self) -> None:
        start = time.perf_counter()
        self._fps_timer = start
        self._setup_glfw_callbacks()
        last_key_state = {}
        _key_repeat_delay = 0.25

        while not glfw.window_should_close(self._window):
            # Keyboard shortcuts (key-down edge detection)
            for key_name, key_val in [
                ("ESCAPE", glfw.KEY_ESCAPE), ("SPACE", glfw.KEY_SPACE),
                ("C", glfw.KEY_C), ("T", glfw.KEY_T),
                ("1", glfw.KEY_1), ("2", glfw.KEY_2),
                ("F", glfw.KEY_F), ("R", glfw.KEY_R)
            ]:
                pressed = glfw.get_key(self._window, key_val) == glfw.PRESS
                was_pressed = last_key_state.get(key_name, False)
                if pressed and not was_pressed:
                    if key_name == "ESCAPE":
                        glfw.set_window_should_close(self._window, True)
                    elif key_name == "SPACE":
                        self._paused = not self._paused
                        if self._paused:
                            self._start_off = time.perf_counter()
                        else:
                            start += time.perf_counter() - self._start_off
                    elif key_name == "C":
                        new_mode = CinematicCamera.MANUAL if self._camera.mode == CinematicCamera.AUTO else CinematicCamera.AUTO
                        if new_mode == CinematicCamera.MANUAL:
                            # Capture current auto values for smooth transition
                            self._camera._auto_yaw = 0.0
                            self._camera._auto_pitch = 0.0
                            self._camera._auto_radius = 4.0
                            self._camera._auto_tx = 0.0
                            self._camera._auto_ty = 0.0
                            self._camera._auto_tz = 0.0
                        self._camera.switch_mode(new_mode, time.perf_counter() - start)
                    elif key_name == "T":
                        self._target_theme = (self._target_theme + 1) % len(THEMES)
                        self._theme_blend = 0.001
                    elif key_name == "1":
                        self._enable_bloom = not self._enable_bloom
                    elif key_name == "2":
                        self._enable_trails = not self._enable_trails
                    elif key_name == "F":
                        self._show_fps = not self._show_fps
                    elif key_name == "R":
                        self._camera.set_manual(0.0, 0.0, 4.0, 0.0, 0.0, 0.0)
                last_key_state[key_name] = pressed

            # Mouse-driven manual camera
            if self._camera.mode != CinematicCamera.AUTO:
                dx = self._mouse_dx
                dy = self._mouse_dy
                if self._mouse_buttons[0]:  # Left drag — orbit
                    yaw = self._camera.manual_yaw - dx * 0.008
                    pitch = self._camera.manual_pitch + dy * 0.008
                    self._camera.set_manual(yaw, pitch, self._camera.manual_radius,
                                            self._camera.manual_tx,
                                            self._camera.manual_ty,
                                            self._camera.manual_tz)
                if self._mouse_buttons[1]:  # Right drag — pan (move target)
                    tx = self._camera.manual_tx + dx * 0.003
                    ty = self._camera.manual_ty - dy * 0.003
                    self._camera.set_manual(self._camera.manual_yaw,
                                            self._camera.manual_pitch,
                                            self._camera.manual_radius,
                                            tx, ty, self._camera.manual_tz)
                self._mouse_dx = 0.0
                self._mouse_dy = 0.0
            if self._scroll_offset != 0.0:
                r = self._camera.manual_radius - self._scroll_offset * 0.5
                self._camera.set_manual(self._camera.manual_yaw,
                                        self._camera.manual_pitch, r,
                                        self._camera.manual_tx,
                                        self._camera.manual_ty,
                                        self._camera.manual_tz)
                self._scroll_offset = 0.0

            elapsed = time.perf_counter() - start
            self.render(elapsed)
            glfw.swap_buffers(self._window)
            glfw.poll_events()
            self._frame_count += 1

            # Build window title
            theme_name = THEME_NAMES[self._target_theme]
            parts = [WINDOW_TITLE]
            if self._show_fps:
                now = time.perf_counter()
                if now - self._fps_timer >= 1.0:
                    self._fps_value = round(self._frame_count / (now - self._fps_timer))
                    self._frame_count = 0
                    self._fps_timer = now
                parts.append(f"{self._fps_value} fps")
            parts.append(f"{self._particle_count} particles")
            parts.append(f"Theme: {theme_name}")
            if self._camera.mode != CinematicCamera.AUTO:
                parts.append("Manual")
            if self._paused:
                parts.append("Paused")
            glfw.set_window_title(self._window, "  |  ".join(parts))

        self.close()

    def _setup_glfw_callbacks(self) -> None:
        """Set up mouse callbacks for interactive camera control."""
        win = self._window

        def mouse_button_cb(w, button, action, mods):
            if button < 3:
                self._mouse_buttons[button] = action == glfw.PRESS
            if action == glfw.PRESS:
                self._mouse_dx = 0.0
                self._mouse_dy = 0.0
                # If in auto mode, switch to manual on click
                if self._camera.mode == CinematicCamera.AUTO:
                    self._camera._auto_yaw = 0.0
                    self._camera._auto_pitch = 0.0
                    self._camera._auto_radius = 4.0
                    self._camera._auto_tx = 0.0
                    self._camera._auto_ty = 0.0
                    self._camera._auto_tz = 0.0
                    self._camera._transition_time = 0.0
                    self._camera.mode = CinematicCamera.MANUAL

        def cursor_pos_cb(w, xpos, ypos):
            if any(self._mouse_buttons):
                self._mouse_dx += xpos - self._mouse_x
                self._mouse_dy += ypos - self._mouse_y
            self._mouse_x = xpos
            self._mouse_y = ypos

        def scroll_cb(w, xoff, yoff):
            self._scroll_offset += yoff

        glfw.set_mouse_button_callback(win, mouse_button_cb)
        glfw.set_cursor_pos_callback(win, cursor_pos_cb)
        glfw.set_scroll_callback(win, scroll_cb)

    def self_test(self) -> None:
        for i in range(3):
            self.render(i * 0.05)
            glfw.swap_buffers(self._window)
            glfw.poll_events()
        self.close()

    def close(self) -> None:
        try:
            for attr in ["_particle_vao", "_vbo", "_prog_particles",
                         "_star_vao", "_star_vbo", "_prog_stars",
                         "_spark_vao", "_spark_vbo", "_prog_sparks",
                         "_ring_vao", "_ring_vbo", "_prog_rings"]:
                obj = getattr(self, attr, None)
                if obj is not None:
                    obj.release()
            if self._bloom is not None:
                self._bloom.release()
            if self._trail is not None:
                self._trail.release()
            self._ctx.release()
        finally:
            if self._window is not None:
                glfw.destroy_window(self._window)
            glfw.terminate()
            self._window = None


# ═══════════════════════════════════════════════════════════════
# Section 10: Entry Point
# ═══════════════════════════════════════════════════════════════

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="3D Particle Heart v2 — Cinematic real-time particle art")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=960)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--particles", type=int, default=70000,
                   help="Total particle count (default 70000, max ~500000)")
    p.add_argument("--bloom", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable HDR bloom post-processing")
    p.add_argument("--trails", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable motion trail accumulation")
    p.add_argument("--fps", action="store_true", help="Show FPS counter in window title")
    p.add_argument("--self-test", action="store_true",
                   help="Headless render test, exit after a few frames")
    p.add_argument("--theme", type=int, default=0, choices=range(len(THEMES)),
                   help=f"Color theme: {', '.join(f'{i}={n}' for i,n in enumerate(THEME_NAMES))}")
    p.add_argument("--quality", type=str, default="medium", choices=["low", "medium", "high"],
                   help="Quality preset: low (fewer particles, faster), medium (balanced), "
                        "high (dense particles, full effects)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # I3: apply quality preset overrides (only when user hasn't explicitly set them)
    import sys as _sys
    has_particles = any(a.startswith("--particles") for a in _sys.argv[1:])
    has_bloom = any(a.startswith("--bloom") for a in _sys.argv[1:])
    has_trails = any(a.startswith("--trails") for a in _sys.argv[1:])
    if args.quality == "low":
        if not has_particles:
            args.particles = 40000
        if not has_bloom:
            args.bloom = True
        if not has_trails:
            args.trails = False
    elif args.quality == "high":
        if not has_particles:
            args.particles = 90000
        if not has_bloom:
            args.bloom = True
        if not has_trails:
            args.trails = True
    app = ParticleHeartV2App(args)
    if args.self_test:
        app.self_test()
        print("Self-test passed.")
    else:
        app.run()


if __name__ == "__main__":
    main()
