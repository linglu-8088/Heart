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

# Default particle counts per layer (scaled by --particles factor)
_BASE_OUTLINE = 40000
_BASE_SHELL = 50000
_BASE_FILL = 120000
_BASE_CORE = 35000
_BASE_ORBIT = 30000
_BASE_TOTAL = _BASE_OUTLINE + _BASE_SHELL + _BASE_FILL + _BASE_CORE + _BASE_ORBIT
_BASE_STARS = 2000
_BASE_SPARKS = 5000
_BASE_RINGS = 2048

# 3D depth scale: max half-thickness at the heart centre, tapers to zero at surface
_HEART_DEPTH = 0.32

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

_HEART_SCALE = 1.0 / 13.0


def heart_boundary(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Classic 2D parametric heart curve.

    Returns (x, y) in approx [-1.23, 1.23] x [-1.31, 0.89].
    Cleft at centre-top, point at bottom.
    """
    x = 16.0 * np.sin(theta) ** 3
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


# ═══════════════════════════════════════════════════════════════
# Section 3: Particle Generation
# ═══════════════════════════════════════════════════════════════

def _lerp_colors(ca: np.ndarray, cb: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = t[:, None]
    return ca + (cb - ca) * t


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
    theta = np.linspace(0.0, math.tau, count, endpoint=False)
    theta += rng.normal(0.0, 0.015, count)
    base_x, base_y = heart_boundary(theta)
    shell = 0.96 + 0.08 * rng.random(count) + rng.normal(0.0, 0.012, count)
    jitter = rng.normal(0.0, 0.016, (count, 2))
    x = base_x * shell + jitter[:, 0]
    y = base_y * shell + jitter[:, 1]
    z = rng.normal(0.0, 0.05, count)

    positions = np.column_stack([x, y, z])
    n = count
    radial = np.clip(0.82 + rng.random(n) * 0.22, 0.78, 1.04)
    mix = rng.random(n)
    colors = _lerp_colors(
        np.array([0.94, 0.14, 0.28], dtype="f4"),
        np.array([0.98, 0.30, 0.46], dtype="f4"),
        mix,
    )
    sizes = rng.uniform(4.0, 6.5, n)
    phase = rng.uniform(0.0, math.tau, n)
    alpha = rng.uniform(0.50, 0.80, n)
    speed = rng.uniform(0.8, 1.3, n)
    return _stack_particles(positions, colors, sizes, phase, 0.0, radial, alpha, speed)


def _sample_shell_new(rng, count):
    theta = rng.uniform(0.0, math.tau, count)
    base_x, base_y = heart_boundary(theta)
    boundary_factor = 0.28 + 0.68 * (rng.random(count) ** 1.2)
    xy_noise = rng.normal(0.0, 0.025, (count, 2))
    x = base_x * boundary_factor + xy_noise[:, 0]
    y = base_y * boundary_factor + xy_noise[:, 1]

    z_max = _HEART_DEPTH * np.power(np.maximum(0.0, 1.0 - boundary_factor), 0.45)
    z = rng.normal(0.0, z_max * 0.42, count)

    positions = np.column_stack([x, y, z])
    n = count
    radial = np.clip(boundary_factor * 0.85 + 0.12, 0.18, 0.98)
    mix = np.clip(0.35 + 0.55 * rng.random(n), 0.0, 1.0)
    colors = _lerp_colors(
        np.array([0.96, 0.20, 0.32], dtype="f4"),
        np.array([0.98, 0.42, 0.52], dtype="f4"),
        mix,
    )
    sizes = rng.uniform(4.5, 7.5, n)
    phase = rng.uniform(0.0, math.tau, n)
    alpha = rng.uniform(0.35, 0.65, n)
    speed = rng.uniform(0.9, 1.6, n)
    return _stack_particles(positions, colors, sizes, phase, 1.0, radial, alpha, speed)


def _sample_fill_new(rng, count):
    theta = rng.uniform(0.0, math.tau, count)
    base_x, base_y = heart_boundary(theta)
    # power 2.5: median bf=0.29 — particles concentrated in deep interior for dense body
    boundary_factor = 0.02 + 0.86 * (rng.random(count) ** 2.5)
    xy_noise_scale = 0.025 + 0.030 * (1.0 - boundary_factor)
    x = base_x * boundary_factor + rng.normal(0.0, xy_noise_scale, count)
    y = base_y * boundary_factor + rng.normal(0.0, xy_noise_scale, count)

    z_max = _HEART_DEPTH * np.power(np.maximum(0.0, 1.0 - boundary_factor), 0.45)
    z = rng.normal(0.0, z_max * 0.44, count)

    positions = np.column_stack([x, y, z])
    n = count
    radial = np.clip(boundary_factor, 0.0, 1.0)
    warm_mix = np.clip(boundary_factor * 0.85 + rng.normal(0.0, 0.08, n), 0.0, 1.0)
    colors = _lerp_colors(
        np.array([0.98, 0.46, 0.38], dtype="f4"),
        np.array([0.94, 0.12, 0.28], dtype="f4"),
        warm_mix,
    )
    colors = _lerp_colors(colors, np.array([0.98, 0.54, 0.48], dtype="f4"),
                          (1.0 - boundary_factor) * 0.12)
    sizes = rng.uniform(4.0, 7.0, n)
    phase = rng.uniform(0.0, math.tau, n)
    alpha = rng.uniform(0.18, 0.38, n)
    speed = rng.uniform(0.7, 1.5, n)
    return _stack_particles(positions, colors, sizes, phase, 2.0, radial, alpha, speed)


def _sample_core_new(rng, count):
    # Uniform sampling in a 3D ellipsoid (slightly squashed in z)
    phi = rng.uniform(0.0, math.pi, count)
    theta = rng.uniform(0.0, math.tau, count)
    r = rng.random(count) ** (1.0 / 3.0) * 0.35
    x = r * np.sin(phi) * np.cos(theta) * 0.95 + rng.normal(0.0, 0.025, count)
    y = r * np.sin(phi) * np.sin(theta) * 1.08 + rng.normal(0.0, 0.025, count)
    z = r * np.cos(phi) * 0.75 + rng.normal(0.0, 0.03, count)

    positions = np.column_stack([x, y, z])
    n = count
    radial = np.clip(r / 0.35, 0.0, 0.32)
    mix = rng.random(n)
    colors = _lerp_colors(
        np.array([0.98, 0.40, 0.52], dtype="f4"),
        np.array([0.99, 0.56, 0.64], dtype="f4"),
        mix,
    )
    sizes = rng.uniform(6.0, 10.0, n)
    phase = rng.uniform(0.0, math.tau, n)
    alpha = rng.uniform(0.35, 0.60, n)
    speed = rng.uniform(1.0, 2.2, n)
    return _stack_particles(positions, colors, sizes, phase, 3.0, radial, alpha, speed)


def _sample_orbit_new(rng, count):
    angle = rng.uniform(0.0, math.tau, count)
    path_choice = rng.random(count)

    s_val = angle
    denom = 1.0 + np.sin(s_val) ** 2
    lem_x = 0.85 * np.cos(s_val) / denom
    lem_y = 0.55 * np.sin(s_val) * np.cos(s_val) / denom
    lem_z = 0.45 * np.sin(s_val) / denom

    r_circ = rng.uniform(1.15, 2.0, count)
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
        sc_r = rng.uniform(1.2, 2.2, sc)
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
        np.array([1.0, 0.25, 0.22], dtype="f4"),
        np.array([1.0, 0.65, 0.35], dtype="f4"),
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
    cw = np.full((count, 3), 0.95, dtype="f4")
    cw[:, 2] = rng.uniform(0.90, 1.00, count)
    phase = rng.uniform(0.0, math.tau, count)
    speed = rng.uniform(0.05, 0.20, count)
    return _stack_particles(np.column_stack([x, y, z]), cw,
                            rng.uniform(1.5, 3.5, count),
                            phase, 5.0, np.zeros(count),
                            np.full(count, 0.7, dtype="f4"),
                            speed)


def _sample_sparks(rng, count):
    theta = rng.uniform(0.0, math.tau, count)
    base_x, base_y = heart_boundary(theta)
    nx, ny = _heart_outward_normal(theta)
    shell = 0.96 + 0.06 * rng.random(count)
    jitter = rng.normal(0.0, 0.014, (count, 2))
    x = base_x * shell + jitter[:, 0]
    y = base_y * shell + jitter[:, 1]
    z = rng.normal(0.0, 0.08, count)

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
    factor = total_count / _BASE_TOTAL
    counts = {
        "outline": max(2000, int(_BASE_OUTLINE * factor)),
        "shell": max(3000, int(_BASE_SHELL * factor)),
        "fill": max(20000, int(_BASE_FILL * factor)),
        "core": max(3000, int(_BASE_CORE * factor)),
        "orbit": max(8000, int(_BASE_ORBIT * factor)),
    }
    adj_total = sum(counts.values())
    if adj_total != total_count:
        counts["fill"] += total_count - adj_total

    print(f"Generating {total_count} particles: "
          f"outline={counts['outline']}, shell={counts['shell']}, "
          f"fill={counts['fill']}, core={counts['core']}, orbit={counts['orbit']}")

    t0 = time.perf_counter()
    layers = [
        _sample_outline_new(rng, counts["outline"]),
        _sample_shell_new(rng, counts["shell"]),
        _sample_fill_new(rng, counts["fill"]),
        _sample_core_new(rng, counts["core"]),
        _sample_orbit_new(rng, counts["orbit"]),
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
uniform vec3 u_theme_tint[5];

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
    float edge_lock = smoothstep(0.78, 1.0, in_radial);
    float core_mix = 1.0 - smoothstep(0.25, 0.95, in_radial);
    float pulse = u_beat;
    float shock = u_shockwave;
    float shimmer = 0.5 + 0.5 * sin(u_time * (1.2 + in_speed * 0.35) + in_phase);

    if (in_kind < 0.5) {
        // Outline: subtle wobble
        pos.xy *= 1.0 + pulse * 0.025;
        pos.z += 0.006 * sin(u_time * 1.8 + in_phase) * (1.0 - edge_lock);
        pos.z += 0.010 * sin(u_time * 0.75 + in_phase * 0.7);
    } else if (in_kind < 1.5) {
        // Shell: gentle breathing
        float yaw = 0.15 * sin(u_time * 0.72);
        float pitch = 0.06 * cos(u_time * 0.40);
        vec3 rot = rot_x(rot_y(pos, yaw), pitch);
        pos = mix(pos, rot, (1.0 - edge_lock) * 0.20);
        pos.z += 0.018 * sin(u_time * (1.05 + in_speed * 0.18) + in_phase) * (0.65 - 0.4 * edge_lock);
        pos.xy *= 1.0 + pulse * 0.020;
    } else if (in_kind < 2.5) {
        // Fill: expansion + swirl + shockwave
        float yaw = 0.28 * sin(u_time * 0.78);
        float pitch = 0.07 * cos(u_time * 0.45);
        vec3 rot = rot_x(rot_y(pos, yaw), pitch);
        pos = mix(pos, rot, (1.0 - edge_lock) * 0.35);
        float swirl = sin(u_time * (1.45 + in_speed * 0.28) + in_phase);
        pos.x += swirl * 0.02 * core_mix;
        pos.z += cos(u_time * 1.18 + in_phase) * 0.04 * core_mix;
        pos.y += sin(u_time * 1.08 + in_phase * 1.2) * 0.012 * core_mix;
        // Shockwave ripple: radial push
        pos *= 1.0 + shock * 0.010 * (1.0 - abs(in_radial - 0.5) * 2.0);
        pos.xy *= 1.0 + pulse * (0.020 + 0.010 * core_mix);
        pos.z *= 1.0 + pulse * 0.07;
    } else if (in_kind < 3.5) {
        // Core: strong pulse + burst
        pos.xy *= 1.0 + pulse * (0.025 + 0.015 * core_mix);
        pos.z *= 1.0 + pulse * 0.08;
        pos *= 1.0 + shock * 0.015 * core_mix;
    } else {
        // Orbit: figure-8 / circular paths
        float orbit = u_time * (0.95 + in_speed * 0.45) + in_phase;
        float c = cos(orbit), s = sin(orbit);
        float denom = 1.0 + s * s;
        // Blend figure-8 with orbital motion based on phase
        float blend = 0.5 + 0.5 * sin(in_phase * 2.7);
        vec3 fig8 = vec3(
            in_position.x + 0.18 * c / denom,
            in_position.y + 0.18 * s * c / denom * 0.7,
            in_position.z + 0.12 * s / denom
        );
        vec3 circ = vec3(
            in_position.x + 0.06 * cos(orbit),
            in_position.y + 0.04 * sin(orbit * 1.2),
            in_position.z + 0.08 * sin(orbit)
        );
        pos = mix(circ, fig8, blend);
        pos *= 1.0 + pulse * 0.04;
    }

    // Camera-space transform
    vec4 view_pos = u_view * vec4(pos, 1.0);
    gl_Position = u_proj * view_pos;

    float depth = -view_pos.z;
    float frontness = saturate(pos.z * 1.15 + 0.5);
    float depth_scale = 1.55 / max(2.0, depth);
    float pulse_size = 1.0 + pulse * mix(0.09, 0.25, core_mix);
    float front_size = mix(0.84, 1.22, frontness);
    gl_PointSize = clamp(in_size * u_point_scale * depth_scale * pulse_size * front_size, 1.2, 35.0);

    // Lighting
    float depth_light = saturate(1.28 - 0.14 * abs(depth));
    float sil_light = mix(0.90, 0.96, edge_lock);
    float shim_light = mix(0.95, 1.04, shimmer);
    vec3 depth_tint = mix(vec3(0.72, 0.78, 0.96), vec3(1.10, 0.98, 0.92), frontness);
    if (in_kind > 3.5) sil_light *= 1.03;
    v_color = in_color * depth_tint * depth_light * sil_light * shim_light * u_theme_tint[int(in_kind)];

    // HDR brightness for bloom extraction
    v_brightness = 1.0
        + pulse * core_mix * 2.5
        + shock * (1.0 - abs(in_radial - shock) * 3.0) * 1.8
        + core_mix * 0.6;

    // Depth fog (mild, to keep particles visible)
    float fog = exp(-depth * 0.08);
    v_alpha = in_alpha * fog * mix(0.85, 1.15, frontness) * mix(0.95, 1.10, shimmer);
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
    float wobble = 0.86
        + 0.06 * sin(angle * 5.0 + v_seed * 1.9)
        + 0.03 * sin(angle * 9.0 - v_seed * 1.3);
    float particle_mask = 1.0 - smoothstep(wobble - 0.20, wobble, radius);
    float dense = exp(-r2 * 7.0);
    float soft = exp(-r2 * 3.0);
    float halo = exp(-r2 * 2.0);

    float alpha_shape = particle_mask * mix(0.62, 1.0, dense);
    float glow = mix(0.95, 1.06, halo);

    if (v_kind < 1.5) {
        alpha_shape *= mix(0.84, 1.0, dense);
        glow = mix(0.96, 1.03, halo);
    } else if (v_kind < 3.5) {
        alpha_shape *= mix(0.86, 1.0, soft);
        glow = mix(0.95, 1.08, halo);
    } else {
        alpha_shape *= mix(0.82, 1.0, soft);
        glow = mix(0.94, 1.10, halo);
    }
    glow *= mix(1.0, 1.06, v_core_mix);

    // HDR color: brightness can exceed 1.0 for bloom extraction
    vec3 hdr = v_color * glow * v_brightness;
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

    // Warm golden-white glow
    vec3 core_gold = vec3(1.0, 0.85, 0.4);
    vec3 tip_white = vec3(1.0, 0.95, 0.8);
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
        self._proj = self._perspective(math.radians(42.0), aspect, 0.1, 20.0)

    def update(self, elapsed: float, aspect: float):
        # Compute auto-cinematic values
        a_yaw = (0.35 * math.sin(0.12 * elapsed + 1.3)
                 + 0.15 * math.sin(0.27 * elapsed + 0.7)
                 + 0.06 * math.sin(0.45 * elapsed + 3.9))
        a_pitch = (0.10 * math.sin(0.15 * elapsed + 2.5)
                   + 0.04 * math.sin(0.38 * elapsed + 0.3)
                   + 0.02 * math.sin(0.52 * elapsed + 1.8))
        a_radius = (3.50 + 0.55 * math.sin(0.08 * elapsed)
                    + 0.22 * math.sin(0.22 * elapsed + 1.2))
        a_tx = 0.05 * math.sin(0.13 * elapsed + 0.6)
        a_ty = 0.04 * math.sin(0.17 * elapsed + 1.5)
        a_tz = 0.02 * math.sin(0.11 * elapsed + 0.8)

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
        phase = (elapsed / self._period) % 1.0
        p1 = math.exp(-((phase - 0.10) / 0.03) ** 2)
        p2 = 0.40 * math.exp(-((phase - 0.25) / 0.05) ** 2)
        p3 = 0.12 * math.exp(-((phase - 0.50) / 0.10) ** 2)
        self.beat = p1 + p2 + p3

        wave = (phase - 0.10) / 0.30
        self.shockwave = max(0.0, min(1.0, wave)) * (1.0 if 0.10 <= phase <= 0.40 else 0.0)
        self.bloom_intensity = 0.35 + 0.55 * self.beat


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
        self.prog_bright["u_threshold"].value = 0.55
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
    DECAY = 0.90

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
        self._u_theme_tint = self._prog_particles["u_theme_tint"]

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
        point_scale = min(w, h) / 80.0

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
        trail_strength = 0.25 if self._trail is not None else 0.0
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
    p.add_argument("--particles", type=int, default=300000,
                   help="Total particle count (default 300000, max ~500000)")
    p.add_argument("--bloom", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable HDR bloom post-processing")
    p.add_argument("--trails", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable motion trail accumulation")
    p.add_argument("--fps", action="store_true", help="Show FPS counter in window title")
    p.add_argument("--self-test", action="store_true",
                   help="Headless render test, exit after a few frames")
    p.add_argument("--theme", type=int, default=0, choices=range(len(THEMES)),
                   help=f"Color theme: {', '.join(f'{i}={n}' for i,n in enumerate(THEME_NAMES))}")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    app = ParticleHeartV2App(args)
    if args.self_test:
        app.self_test()
        print("Self-test passed.")
    else:
        app.run()


if __name__ == "__main__":
    main()
