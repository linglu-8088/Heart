from __future__ import annotations

import math
import time

import numpy as np

from .constants import PARTICLE_RATIOS
from .geometry import (
    HEART_DEPTH,
    _heart_local_radius,
    _heart_outward_normal,
    _z_depth,
    heart_boundary,
)


def _lerp_colors(ca: np.ndarray, cb: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t[:, None]
    return ca + (cb - ca) * t


def _gradient_3stop(
    t: np.ndarray,
    c0: np.ndarray,
    c1: np.ndarray,
    c2: np.ndarray,
    t1: float = 0.45,
) -> np.ndarray:
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
    return np.hstack(
        [
            positions.astype("f4"),
            colors.astype("f4"),
            sizes[:, None].astype("f4"),
            phase[:, None].astype("f4"),
            kind_col,
            distance[:, None].astype("f4"),
            alpha[:, None].astype("f4"),
            speed[:, None].astype("f4"),
        ]
    )


def _sample_outline_new(rng, count):
    n_uniform = int(count * 0.72)
    n_tip = count - n_uniform
    theta_u = rng.uniform(0.0, math.tau, n_uniform)
    tip_raw = rng.beta(3.5, 3.5, n_tip)
    theta_tip = 0.7 * math.pi + (1.3 * math.pi - 0.7 * math.pi) * tip_raw
    theta = np.concatenate([theta_u, theta_tip])
    theta += rng.normal(0.0, 0.008, count)
    base_x, base_y = heart_boundary(theta)
    shell = 0.985 + 0.020 * rng.random(count) + rng.normal(0.0, 0.006, count)
    jitter = rng.normal(0.0, 0.010, (count, 2))
    x = base_x * shell + jitter[:, 0]
    y = base_y * shell + jitter[:, 1]

    r_factor = np.clip(0.82 + rng.random(count) * 0.22, 0.78, 1.04)
    z = _z_depth(r_factor, theta, count, rng)

    positions = np.column_stack([x, y, z])
    n = count
    deep_red = np.array([0.76, 0.09, 0.36], dtype="f4")
    bright_red = np.array([1.00, 0.18, 0.33], dtype="f4")
    light_gold = np.array([1.00, 0.72, 0.40], dtype="f4")
    nr = np.clip((r_factor - 0.78) / 0.26, 0.0, 1.0)
    nr += rng.normal(0.0, 0.10, n)
    nr = np.clip(nr, 0.0, 1.0)
    colors = _gradient_3stop(1.0 - nr, deep_red, bright_red, light_gold, 0.35)
    size_factor = 0.60 + 0.40 * (1.0 - (r_factor - 0.78) / 0.26)
    sizes = rng.uniform(2.0, 4.0, n) * size_factor
    phase = rng.uniform(0.0, math.tau, n)
    base_alpha = rng.uniform(0.66, 0.92, n)
    alpha_factor = 0.72 + 0.28 * (1.0 - (r_factor - 0.78) / 0.26)
    alpha = base_alpha * alpha_factor
    speed = rng.uniform(0.8, 1.3, n)
    return _stack_particles(positions, colors, sizes, phase, 0.0, r_factor, alpha, speed)


def _sample_loose_outline(rng, count):
    theta = rng.uniform(0.0, math.tau, count)
    theta += rng.normal(0.0, 0.015, count)
    base_x, base_y = heart_boundary(theta)
    shell = 1.006 + 0.016 * rng.random(count) + rng.normal(0.0, 0.006, count)
    jitter = rng.normal(0.0, 0.008, (count, 2))
    x = base_x * shell + jitter[:, 0]
    y = base_y * shell + jitter[:, 1]

    r_factor = np.clip(0.90 + rng.random(count) * 0.12, 0.88, 1.04)
    z = rng.normal(0.0, HEART_DEPTH * 0.06, count)

    positions = np.column_stack([x, y, z])
    n = count
    dr = np.array([0.62, 0.06, 0.28], dtype="f4")
    cr = np.array([0.40, 0.03, 0.18], dtype="f4")
    nr = np.clip(rng.normal(0.4, 0.25, n), 0.0, 1.0)
    colors = _lerp_colors(dr, cr, nr)
    sizes = rng.uniform(1.0, 2.5, n)
    phase = rng.uniform(0.0, math.tau, n)
    alpha = rng.uniform(0.03, 0.08, n)
    speed = rng.uniform(0.6, 1.1, n)
    return _stack_particles(positions, colors, sizes, phase, 0.3, r_factor, alpha, speed)


def _sample_shell_new(rng, count):
    theta = rng.uniform(0.0, math.tau, count)
    base_x, base_y = heart_boundary(theta)
    r_factor = 0.55 + 0.40 * (rng.random(count) ** 0.55)
    xy_noise = rng.normal(0.0, 0.020, (count, 2))
    x = base_x * r_factor + xy_noise[:, 0]
    y = base_y * r_factor + xy_noise[:, 1]
    z = _z_depth(r_factor, theta, count, rng)

    positions = np.column_stack([x, y, z])
    n = count
    deep_red = np.array([0.78, 0.10, 0.34], dtype="f4")
    bright_red = np.array([1.00, 0.22, 0.37], dtype="f4")
    light_gold = np.array([1.00, 0.72, 0.40], dtype="f4")
    nr = np.clip(r_factor / 0.95, 0.0, 1.0)
    nr += rng.normal(0.0, 0.10, n)
    nr = np.clip(nr, 0.0, 1.0)
    colors = _gradient_3stop(1.0 - nr, deep_red, bright_red, light_gold, 0.38)
    size_factor = 0.70 + 0.30 * (1.0 - r_factor / 0.95)
    sizes = rng.uniform(2.0, 4.5, n) * size_factor
    phase = rng.uniform(0.0, math.tau, n)
    base_alpha = rng.uniform(0.42, 0.72, n)
    alpha_factor = 0.72 + 0.28 * (1.0 - r_factor / 0.95)
    alpha = base_alpha * alpha_factor
    speed = rng.uniform(0.9, 1.6, n)
    return _stack_particles(positions, colors, sizes, phase, 1.0, r_factor, alpha, speed)


def _sample_fill_new(rng, count):
    oversample = int(count * 1.25)
    theta = rng.uniform(0.0, math.tau, oversample)
    base_x, base_y = heart_boundary(theta)
    r_factor = 0.02 + 0.92 * (rng.random(oversample) ** 1.4)
    xy_noise_scale = 0.025 + 0.035 * (1.0 - r_factor)
    x = base_x * r_factor + rng.normal(0.0, xy_noise_scale, oversample)
    y = base_y * r_factor + rng.normal(0.0, xy_noise_scale, oversample)
    z = _z_depth(r_factor, theta, oversample, rng)

    positions = np.column_stack([x, y, z])
    edge_mask = r_factor > 0.70
    survival = np.ones(oversample)
    if edge_mask.any():
        survival[edge_mask] = 1.0 - (r_factor[edge_mask] - 0.70) / 0.22
        survival[edge_mask] = np.clip(survival[edge_mask], 0.03, 1.0)
    keep = rng.random(oversample) < survival
    kept_idx = np.where(keep)[0]
    if len(kept_idx) > count:
        kept_idx = rng.choice(kept_idx, count, replace=False)
    elif len(kept_idx) < count:
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
    deep_red = np.array([0.76, 0.09, 0.36], dtype="f4")
    bright_red = np.array([1.00, 0.18, 0.33], dtype="f4")
    light_gold = np.array([1.00, 0.72, 0.40], dtype="f4")
    nr = np.clip(r_factor / 0.94, 0.0, 1.0)
    nr += rng.normal(0.0, 0.08, n)
    nr = np.clip(nr, 0.0, 1.0)
    colors = _gradient_3stop(1.0 - nr, deep_red, bright_red, light_gold, 0.40)
    size_factor = 0.55 + 0.45 * (1.0 - np.clip(r_factor / 0.88, 0.0, 1.0))
    sizes = rng.uniform(2.0, 3.5, n) * size_factor
    phase = rng.uniform(0.0, math.tau, n)
    base_alpha = rng.uniform(0.28, 0.50, n)
    alpha_factor = 0.55 + 0.45 * (1.0 - np.clip(r_factor / 0.88, 0.0, 1.0))
    alpha = base_alpha * alpha_factor
    speed = rng.uniform(0.7, 1.5, n)
    return _stack_particles(positions, colors, sizes, phase, 2.0, r_factor, alpha, speed)


def _sample_core_new(rng, count):
    theta = rng.uniform(0.0, math.tau, count)
    phi = rng.uniform(0.0, math.pi, count)
    r = rng.random(count) ** (1.0 / 3.0) * 0.35
    r_factor = r / 0.35
    local_r = _heart_local_radius(theta)
    heart_scale = np.clip(local_r / 1.17, 0.50, 1.0)
    hx = r * np.sin(phi) * np.cos(theta) * 0.95
    hy = r * np.sin(phi) * np.sin(theta) * 1.08
    x = hx * heart_scale + rng.normal(0.0, 0.020, count)
    y = hy * heart_scale + rng.normal(0.0, 0.020, count)
    z = r * np.cos(phi) * 0.75 + rng.normal(0.0, 0.03, count)

    positions = np.column_stack([x, y, z])
    n = count
    br = np.array([1.00, 0.69, 0.75], dtype="f4")
    dr = np.array([1.00, 0.30, 0.43], dtype="f4")
    nr = np.clip(r_factor, 0.0, 1.0)
    nr += rng.normal(0.0, 0.10, n)
    nr = np.clip(nr, 0.0, 1.0)
    colors = _lerp_colors(br, dr, nr)
    size_factor = 0.78 + 0.22 * (1.0 - r_factor)
    sizes = rng.uniform(3.0, 5.5, n) * size_factor
    phase = rng.uniform(0.0, math.tau, n)
    alpha = rng.uniform(0.35, 0.55, n)
    speed = rng.uniform(1.0, 2.2, n)
    return _stack_particles(positions, colors, sizes, phase, 3.0, r_factor, alpha, speed)


def _sample_trail_orbit(rng, count):
    theta = rng.uniform(0.0, math.tau, count)
    base_x, base_y = heart_boundary(theta)
    shell = 1.06 + rng.random(count) * 0.15
    jitter = rng.normal(0.0, 0.008, (count, 2))
    x = base_x * shell + jitter[:, 0]
    y = base_y * shell + jitter[:, 1]
    z = rng.normal(0.0, HEART_DEPTH * 0.15, count)

    positions = np.column_stack([x, y, z])
    n = count
    gold = np.array([1.00, 0.85, 0.35], dtype="f4")
    white = np.array([1.00, 0.95, 0.70], dtype="f4")
    mix = rng.random(n)
    colors = _lerp_colors(gold, white, mix)
    sizes = rng.uniform(0.8, 2.0, n)
    phase = rng.uniform(0.0, math.tau, n)
    alpha = rng.uniform(0.08, 0.18, n)
    speed = rng.uniform(0.25, 0.65, n)
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
    radial = np.full(n, 0.15, dtype="f4")
    mix = np.clip(rng.normal(0.5, 0.22, n), 0.0, 1.0)
    colors = _lerp_colors(
        np.array([1.0, 0.60, 0.15], dtype="f4"),
        np.array([1.0, 0.85, 0.30], dtype="f4"),
        mix,
    )
    sizes = rng.uniform(1.2, 3.0, n)
    phase = rng.uniform(0.0, math.tau, n)
    alpha = rng.uniform(0.06, 0.14, n)
    speed = rng.uniform(0.8, 1.9, n)
    return _stack_particles(positions, colors, sizes, phase, 4.0, radial, alpha, speed)


def _sample_stars(rng, count):
    phi = rng.uniform(0.0, math.pi, count)
    theta = rng.uniform(0.0, math.tau, count)
    radius = rng.uniform(10.0, 20.0, count)
    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)
    cw = np.full((count, 3), 0.95, dtype="f4")
    cw[:, 0] = rng.uniform(0.80, 1.00, count)
    cw[:, 1] = rng.uniform(0.88, 1.00, count)
    cw[:, 2] = rng.uniform(0.92, 1.00, count)
    phase = rng.uniform(0.0, math.tau, count)
    speed = rng.uniform(0.03, 0.25, count)
    return _stack_particles(
        np.column_stack([x, y, z]),
        cw,
        rng.uniform(1.5, 3.5, count),
        phase,
        5.0,
        np.zeros(count),
        np.full(count, 0.7, dtype="f4"),
        speed,
    )


def _sample_sparks(rng, count):
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
    z = rng.normal(0.0, HEART_DEPTH * 0.12, count)

    n = count
    dx = nx + rng.normal(0.0, 0.40, n)
    dy = ny + rng.normal(0.0, 0.40, n)
    dz = rng.normal(0.0, 0.40, n)
    dn = np.maximum(np.sqrt(dx * dx + dy * dy + dz * dz), 1e-8)
    dx /= dn
    dy /= dn
    dz /= dn

    spark_data = np.zeros(
        n,
        dtype=[
            ("pos", "f4", 3),
            ("dir", "f4", 3),
            ("speed", "f4"),
            ("p_off", "f4"),
            ("sz", "f4"),
            ("al", "f4"),
        ],
    )
    spark_data["pos"] = np.column_stack([x, y, z]).astype("f4")
    spark_data["dir"] = np.column_stack([dx, dy, dz]).astype("f4")
    spark_data["speed"] = rng.uniform(0.3, 1.0, n).astype("f4")
    spark_data["p_off"] = rng.uniform(0.0, 0.15, n).astype("f4")
    spark_data["sz"] = rng.uniform(1.5, 4.0, n).astype("f4")
    spark_data["al"] = rng.uniform(0.5, 0.9, n).astype("f4")
    return spark_data


def build_particle_cloud(seed, total_count):
    rng = np.random.default_rng(seed)
    ratios = dict(PARTICLE_RATIOS)
    raw_counts = {key: int(total_count * value) for key, value in ratios.items()}
    raw_counts["fill"] = total_count - sum(value for key, value in raw_counts.items() if key != "fill")
    counts = raw_counts

    print(
        f"Generating {total_count} particles: "
        f"outline={counts['outline']}, loose_outline={counts['loose_outline']}, "
        f"shell={counts['shell']}, fill={counts['fill']}, "
        f"core={counts['core']}, orbit={counts['orbit']}, "
        f"trail_orbit={counts['trail_orbit']}",
    )

    start = time.perf_counter()
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
    end = time.perf_counter()
    print(
        f"Particle generation: {end - start:.2f}s, "
        f"VBO size: {particles.nbytes / 1024 / 1024:.1f} MB",
    )
    return particles


def _sample_pedestal(rng, count):
    _theta_scan = np.linspace(0.0, math.tau, 512, endpoint=False)
    _bx, _by = heart_boundary(_theta_scan)
    _max_abs_x = float(np.max(np.abs(_bx)))
    _min_y = float(np.min(_by))

    # --- Golden crystal pedestal with decorative edges and soft halo ---
    # Circular crystal base with internal glow, decorative rim ring,
    # and a soft ground halo that naturally bridges into the dark background.
    count_core = int(count * 0.30)
    count_rim = int(count * 0.22)
    count_body = int(count * 0.28)
    count_skirt = count - count_core - count_rim - count_body

    R = _max_abs_x * 1.04
    top_core_y = _min_y + 0.010
    top_rim_y = _min_y - 0.003
    body_y_lo = _min_y - 0.160
    body_y_hi = _min_y - 0.008
    skirt_y = _min_y - 0.200

    # --- core: dense circular crystal face under the heart ---
    core_theta = rng.uniform(0.0, math.tau, count_core)
    core_r = rng.random(count_core) ** 0.50
    core_x = core_r * np.cos(core_theta) * R * 0.68
    core_z = core_r * np.sin(core_theta) * R * 0.68
    core_y = top_core_y + rng.uniform(-0.005, 0.005, count_core)
    core_dist = np.sqrt(
        (core_x / (R * 0.68 + 1e-8)) ** 2
        + (core_z / (R * 0.68 + 1e-8)) ** 2
    )
    core_alpha = np.clip(
        0.65 + 0.28 * (1.0 - core_dist) + rng.normal(0.0, 0.018, count_core),
        0.62, 0.95,
    )
    core_sizes = rng.uniform(9.0, 13.0, count_core) * (
        0.94 + 0.06 * (1.0 - core_dist)
    )

    # --- rim: decorative golden ring with gems (bimodal: thin ring + sparkle burst) ---
    n_ring = int(count_rim * 0.60)
    n_gem = count_rim - n_ring
    # Tight ring band
    rim_ring_theta = rng.uniform(0.0, math.tau, n_ring)
    rim_ring_r = 0.92 + 0.10 * rng.random(n_ring)
    rim_ring_x = rim_ring_r * np.cos(rim_ring_theta) * R
    rim_ring_z = rim_ring_r * np.sin(rim_ring_theta) * R
    rim_ring_y = top_rim_y + rng.uniform(-0.004, 0.004, n_ring)
    # Gem sparkle bursts scattered along the rim
    rim_gem_theta = rng.uniform(0.0, math.tau, n_gem)
    rim_gem_r = 0.78 + 0.25 * rng.random(n_gem)
    rim_gem_x = rim_gem_r * np.cos(rim_gem_theta) * R
    rim_gem_z = rim_gem_r * np.sin(rim_gem_theta) * R
    rim_gem_y = top_rim_y + rng.uniform(-0.006, 0.006, n_gem)

    rim_x = np.concatenate([rim_ring_x, rim_gem_x])
    rim_z = np.concatenate([rim_ring_z, rim_gem_z])
    rim_y = np.concatenate([rim_ring_y, rim_gem_y])
    rim_dist = np.sqrt(
        (rim_x / (R + 1e-8)) ** 2
        + (rim_z / (R + 1e-8)) ** 2
    )
    rim_edge = np.clip((rim_dist - 0.75) / 0.30, 0.0, 1.0)
    rim_alpha = np.clip(
        0.58 + 0.30 * (1.0 - rim_edge) + rng.normal(0.0, 0.018, count_rim),
        0.55, 0.90,
    )
    rim_sizes = rng.uniform(8.2, 12.0, count_rim) * (
        0.90 + 0.10 * (1.0 - rim_edge)
    )

    # --- body: crystal volume with hourglass silhouette ---
    body_y = rng.uniform(body_y_lo, body_y_hi, count_body)
    body_t = (body_y - body_y_lo) / (body_y_hi - body_y_lo + 1e-8)
    body_r = rng.random(count_body) ** 0.80
    body_a = rng.uniform(0.0, math.tau, count_body)
    # Hourglass: 72% at top, 52% at mid, 65% at base
    hourglass = 0.72 - 0.20 * np.sin(body_t * math.pi) + 0.13 * (1.0 - body_t)
    body_x = body_r * np.cos(body_a) * R * hourglass
    body_z = body_r * np.sin(body_a) * R * hourglass
    body_dist = np.sqrt(
        (body_x / (R * hourglass + 1e-8)) ** 2
        + (body_z / (R * hourglass + 1e-8)) ** 2
    )
    body_alpha = np.clip(
        0.42 + 0.25 * (1.0 - body_dist) + rng.normal(0.0, 0.018, count_body),
        0.38, 0.68,
    )
    body_sizes = rng.uniform(7.6, 11.0, count_body) * (
        0.86 + 0.14 * (1.0 - body_dist)
    )

    # --- skirt: wide golden halo glow on the ground plane ---
    skirt_r = rng.random(count_skirt) ** 0.50
    skirt_theta = rng.uniform(0.0, math.tau, count_skirt)
    skirt_x = skirt_r * np.cos(skirt_theta) * R * 1.20
    skirt_z = skirt_r * np.sin(skirt_theta) * R * 1.20
    skirt_yv = skirt_y + rng.uniform(-0.008, 0.006, count_skirt)
    skirt_dist = np.sqrt(
        (skirt_x / (R * 1.20 + 1e-8)) ** 2
        + (skirt_z / (R * 1.20 + 1e-8)) ** 2
    )
    skirt_alpha = np.clip(
        0.18 + 0.25 * (1.0 - skirt_dist) + rng.normal(0.0, 0.012, count_skirt),
        0.14, 0.42,
    )
    skirt_sizes = rng.uniform(5.2, 7.8, count_skirt) * (
        0.80 + 0.20 * (1.0 - skirt_dist)
    )

    x = np.concatenate([core_x, rim_x, body_x, skirt_x])
    y = np.concatenate([core_y, rim_y, body_y, skirt_yv])
    z = np.concatenate([core_z, rim_z, body_z, skirt_z])
    sizes = np.concatenate([core_sizes, rim_sizes, body_sizes, skirt_sizes])
    phases = rng.uniform(0.0, math.tau, count)
    layers = np.concatenate([
        np.full(count_core, 0.0),
        np.full(count_rim, 1.0),
        np.full(count_body, 2.0),
        np.full(count_skirt, 3.0),
    ])
    alphas = np.concatenate([core_alpha, rim_alpha, body_alpha, skirt_alpha])

    result = np.column_stack([
        x.astype("f4"),
        y.astype("f4"),
        z.astype("f4"),
        sizes.astype("f4"),
        phases.astype("f4"),
        layers.astype("f4"),
        alphas.astype("f4"),
    ])
    rng.shuffle(result)
    return result[:count]
