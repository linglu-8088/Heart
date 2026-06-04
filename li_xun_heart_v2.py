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
HEART_SCALE = 1.0 / 18.0  # retained for orbit curve reference only

# Default particle counts per layer (scaled by --particles factor)
_BASE_OUTLINE = 15000
_BASE_SHELL = 25000
_BASE_FILL = 100000
_BASE_CORE = 20000
_BASE_ORBIT = 40000
_BASE_TOTAL = _BASE_OUTLINE + _BASE_SHELL + _BASE_FILL + _BASE_CORE + _BASE_ORBIT


# ═══════════════════════════════════════════════════════════════
# Section 1: 3D Noise System
# ═══════════════════════════════════════════════════════════════

class BlinnNoise3D:
    """3D value noise with Perlin improved permutation table + FBM."""

    def __init__(self, seed: int = 0) -> None:
        rng = np.random.default_rng(seed)
        self._perm = np.arange(256, dtype=np.int32)
        rng.shuffle(self._perm)
        self._perm = np.tile(self._perm, 2)

    def _hash_vec(self, xi: np.ndarray, yi: np.ndarray, zi: np.ndarray) -> np.ndarray:
        h = self._perm[self._perm[self._perm[xi & 255] + (yi & 255)] + (zi & 255)]
        return h.astype(np.float64) / 127.5 - 1.0

    @staticmethod
    def _fade(t: np.ndarray) -> np.ndarray:
        return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)

    def noise(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        xi = np.floor(x).astype(np.int32)
        yi = np.floor(y).astype(np.int32)
        zi = np.floor(z).astype(np.int32)
        fx = self._fade(x - xi)
        fy = self._fade(y - yi)
        fz = self._fade(z - zi)

        c000 = self._hash_vec(xi, yi, zi)
        c100 = self._hash_vec(xi + 1, yi, zi)
        c010 = self._hash_vec(xi, yi + 1, zi)
        c110 = self._hash_vec(xi + 1, yi + 1, zi)
        c001 = self._hash_vec(xi, yi, zi + 1)
        c101 = self._hash_vec(xi + 1, yi, zi + 1)
        c011 = self._hash_vec(xi, yi + 1, zi + 1)
        c111 = self._hash_vec(xi + 1, yi + 1, zi + 1)

        x00 = c000 + fx * (c100 - c000)
        x10 = c010 + fx * (c110 - c010)
        x01 = c001 + fx * (c101 - c001)
        x11 = c011 + fx * (c111 - c011)
        y0 = x00 + fy * (x10 - x00)
        y1 = x01 + fy * (x11 - x01)
        return y0 + fz * (y1 - y0)

    def fbm(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
            octaves: int = 4, lacunarity: float = 2.0, gain: float = 0.5) -> np.ndarray:
        value = np.zeros_like(x, dtype=np.float64)
        amplitude = 1.0
        frequency = 1.0
        max_value = 0.0
        for _ in range(octaves):
            value += amplitude * self.noise(x * frequency, y * frequency, z * frequency)
            max_value += amplitude
            amplitude *= gain
            frequency *= lacunarity
        return value / max_value


# ═══════════════════════════════════════════════════════════════
# Section 2: Heart Implicit Surface (Taubin, remapped y-up)
# ═══════════════════════════════════════════════════════════════

# The Taubin surface in its natural frame (cleft at +z, point at -z):
#   F(X,Y,Z) = (X² + 9Y²/4 + Z² − 1)³ − X²Z³ − 9Y²Z³/80
#
# We rotate -90° around X so that Y becomes the up axis:
#   Taubin (X, Y, Z)  ←  Renderer (x, z, -y)
#
# Substituting: X=x, Y=z, Z=-y gives us F in renderer coords.


def _heart_f(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Implicit function value. f<0 = interior, f=0 = surface, f>0 = exterior.

    Taubin surface: F(X,Y,Z) = (X² + 9Y²/4 + Z² − 1)³ − X²Z³ − 9Y²Z³/80
    Mapped to renderer via (X,Y,Z) = (x, -z, y) so that y is up (cleft at +y).
    """
    x2 = x * x
    y2 = y * y
    z2 = z * z
    A = x2 + 2.25 * z2 + y2 - 1.0
    return A * A * A - x2 * y2 * y - (9.0 / 80.0) * z2 * y2 * y


def _heart_gradient(x: np.ndarray, y: np.ndarray, z: np.ndarray):
    """Analytic gradient of f in renderer coordinates (x-right, y-up, z-depth)."""
    x2 = x * x
    y2 = y * y
    z2 = z * z
    y3 = y2 * y
    A = x2 + 2.25 * z2 + y2 - 1.0
    A2 = A * A

    gx = 6.0 * x * A2 - 2.0 * x * y3
    gy = 6.0 * y * A2 - 3.0 * x2 * y2 - (27.0 / 80.0) * z2 * y2
    gz = 13.5 * z * A2 - (9.0 / 40.0) * z * y3

    return gx, gy, gz


def heart_distance(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Signed distance approximation: f / |grad f|. Negative = interior."""
    f_val = _heart_f(x, y, z)
    gx, gy, gz = _heart_gradient(x, y, z)
    grad_norm = np.sqrt(gx * gx + gy * gy + gz * gz)
    grad_norm = np.maximum(grad_norm, 1e-8)
    raw = f_val / grad_norm
    raw = np.clip(raw, -2.0, 2.0)
    return raw


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


def _rejection_sample_volume(rng, noise, count, max_dist, threshold, max_trials_factor=8):
    """Rejection-sample points inside the 3D heart volume with noise-modulated density."""
    bounds = np.array([
        [-1.30, 1.30],  # x
        [-1.30, 1.30],  # y (up, cleft at +y, point at -y)
        [-0.50, 0.50],  # z (depth, thin)
    ])
    chunks = []
    total_trials = 0
    max_trials = count * max_trials_factor
    collected = 0

    while collected < count and total_trials < max_trials:
        batch_count = min(count * 3, 60000)
        total_trials += batch_count
        cx = rng.uniform(bounds[0, 0], bounds[0, 1], batch_count)
        cy = rng.uniform(bounds[1, 0], bounds[1, 1], batch_count)
        cz = rng.uniform(bounds[2, 0], bounds[2, 1], batch_count)

        dist = heart_distance(cx, cy, cz)
        noise_val = noise.fbm(cx * 3.5, cy * 3.5, cz * 3.5, octaves=4)
        density = 0.5 + 0.5 * noise_val

        mask = (dist < max_dist) & (density > threshold)
        if np.any(mask):
            chunks.append(np.column_stack([
                cx[mask], cy[mask], cz[mask], dist[mask], density[mask]
            ]))
            collected += mask.sum()

    if not chunks:
        raise RuntimeError(f"Failed to sample any particles (max_dist={max_dist}, threshold={threshold})")
    result = np.vstack(chunks)[:count]
    return result[:, :3], result[:, 3], result[:, 4]


def _sample_outline_new(rng, noise, count):
    positions, distances, _density = _rejection_sample_volume(
        rng, noise, count, max_dist=0.0, threshold=-0.6, max_trials_factor=30
    )
    # Newton-projection to surface
    for _ in range(3):
        gx, gy, gz = _heart_gradient(positions[:, 0], positions[:, 1], positions[:, 2])
        f_val = _heart_f(positions[:, 0], positions[:, 1], positions[:, 2])
        gnorm2 = gx * gx + gy * gy + gz * gz
        gnorm2 = np.maximum(gnorm2, 1e-8)
        step = f_val / gnorm2
        positions[:, 0] -= gx * step
        positions[:, 1] -= gy * step
        positions[:, 2] -= gz * step

    positions[:, 2] += rng.normal(0.0, 0.008, count)
    distances = heart_distance(positions[:, 0], positions[:, 1], positions[:, 2])
    # radial: 0=deep interior, 1=edge. Outline particles are on the surface → near 1.
    radial = np.clip(0.82 + rng.random(count) * 0.22, 0.78, 1.04)

    mix = rng.random(count)
    colors = _lerp_colors(
        np.array([0.94, 0.14, 0.28], dtype="f4"),
        np.array([0.98, 0.30, 0.46], dtype="f4"),
        mix,
    )
    sizes = rng.uniform(2.9, 5.0, count)
    phase = rng.uniform(0.0, math.tau, count)
    alpha = rng.uniform(0.18, 0.34, count)
    speed = rng.uniform(0.8, 1.3, count)
    return _stack_particles(positions, colors, sizes, phase, 0.0, radial, alpha, speed)


def _sample_shell_new(rng, noise, count):
    positions, distances, _density = _rejection_sample_volume(
        rng, noise, count, max_dist=0.08, threshold=-0.35, max_trials_factor=20
    )
    # Shell particles are near surface: 0.01 < |d| < 0.08, mapping to ~0.60-0.90
    radial = np.clip(0.58 + abs(distances) / 0.09, 0.55, 0.92)
    mix = np.clip(0.35 + 0.55 * rng.random(count), 0.0, 1.0)
    colors = _lerp_colors(
        np.array([0.96, 0.20, 0.32], dtype="f4"),
        np.array([0.98, 0.42, 0.52], dtype="f4"),
        mix,
    )
    sizes = rng.uniform(2.5, 4.8, count)
    phase = rng.uniform(0.0, math.tau, count)
    alpha = rng.uniform(0.08, 0.16, count)
    speed = rng.uniform(0.9, 1.6, count)
    return _stack_particles(positions, colors, sizes, phase, 1.0, radial, alpha, speed)


def _temperature_color(distance, mix_rng=None):
    """Temperature gradient: core hot (white-pink) → edge cool (dark red) → orbit (golden)."""
    temp = 1.0 / (1.0 + np.exp((distance + 0.1) * 8.0))
    if mix_rng is not None:
        temp = np.clip(temp + mix_rng * 0.06, 0.0, 1.0)

    stops_t = np.array([0.0, 0.25, 0.55, 0.80, 1.0], dtype="f4")
    stops_c = np.array([
        [1.00, 0.55, 0.05],   # golden-orange
        [0.55, 0.05, 0.10],   # dark red
        [0.95, 0.15, 0.25],   # deep red
        [1.00, 0.38, 0.48],   # pink
        [1.00, 0.88, 0.88],   # white-pink
    ], dtype="f4")

    idx = np.searchsorted(stops_t, temp, side="right") - 1
    idx = np.clip(idx, 0, len(stops_t) - 2)
    t_local = (temp - stops_t[idx]) / (stops_t[idx + 1] - stops_t[idx] + 1e-8)
    t_local = np.clip(t_local, 0.0, 1.0)
    return stops_c[idx] + (stops_c[idx + 1] - stops_c[idx]) * t_local[:, None]


def _sample_fill_new(rng, noise, count):
    positions, distances, density = _rejection_sample_volume(
        rng, noise, count, max_dist=-0.02, threshold=-0.30, max_trials_factor=6
    )
    # radial: 0=deep interior, 1=edge. Fill ranges from -0.28 (deep) to -0.02 (near surface).
    radial = np.clip(1.0 + distances / 0.28, 0.0, 1.0)
    colors = _temperature_color(distances, rng.normal(0.0, 0.03, count))
    sizes = rng.uniform(1.8, 4.0, count)
    phase = rng.uniform(0.0, math.tau, count)
    alpha = rng.uniform(0.06, 0.15, count)
    speed = rng.uniform(0.7, 1.5, count)
    return _stack_particles(positions, colors, sizes, phase, 2.0, radial, alpha, speed)


def _sample_core_new(rng, noise, count):
    positions, distances, density = _rejection_sample_volume(
        rng, noise, count, max_dist=-0.20, threshold=-0.45, max_trials_factor=10
    )
    # radial: 0=deep interior, 1=edge. Core ranges from -0.38 (deepest) to -0.20 (boundary).
    radial = np.clip(1.0 + distances / 0.38, 0.0, 1.0)
    colors = _temperature_color(distances, rng.normal(0.0, 0.04, count))
    sizes = rng.uniform(4.0, 7.5, count)
    phase = rng.uniform(0.0, math.tau, count)
    alpha = rng.uniform(0.05, 0.10, count)
    speed = rng.uniform(1.0, 2.2, count)
    return _stack_particles(positions, colors, sizes, phase, 3.0, radial, alpha, speed)


def _sample_orbit_new(rng, count):
    angle = rng.uniform(0.0, math.tau, count)
    # Mix of figure-8 seeds, circular seeds, and scattered seeds
    path_choice = rng.random(count)

    # Lemniscate figure-8 seed positions
    s_val = angle
    denom = 1.0 + np.sin(s_val) ** 2
    lem_x = 0.8 * np.cos(s_val) / denom
    lem_y = 0.55 * np.sin(s_val) * np.cos(s_val) / denom
    lem_z = 0.4 * np.sin(s_val) / denom

    # Circular orbit seeds at various radii
    r_circ = rng.uniform(1.05, 1.8, count)
    circ_x = r_circ * np.cos(angle)
    circ_y = rng.normal(0.0, 0.35, count)
    circ_z = r_circ * 0.6 * np.sin(angle)

    # Blend
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
    distances = np.ones(count, dtype="f4") * 0.50
    radial = np.full(count, 0.15, dtype="f4")
    mix = np.clip(rng.normal(0.5, 0.22, count), 0.0, 1.0)
    colors = _lerp_colors(
        np.array([1.0, 0.25, 0.22], dtype="f4"),
        np.array([1.0, 0.65, 0.35], dtype="f4"),
        mix,
    )
    sizes = rng.uniform(1.5, 4.0, count)
    phase = rng.uniform(0.0, math.tau, count)
    alpha = rng.uniform(0.04, 0.09, count)
    speed = rng.uniform(0.8, 1.9, count)
    return _stack_particles(positions, colors, sizes, phase, 4.0, radial, alpha, speed)


def build_particle_cloud(seed, noise, total_count):
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
        _sample_outline_new(rng, noise, counts["outline"]),
        _sample_shell_new(rng, noise, counts["shell"]),
        _sample_fill_new(rng, noise, counts["fill"]),
        _sample_core_new(rng, noise, counts["core"]),
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
        pos.xy *= 1.0 + pulse * 0.055;
        pos.z += 0.015 * sin(u_time * 1.8 + in_phase) * (1.0 - edge_lock);
        pos.z += 0.024 * sin(u_time * 0.75 + in_phase * 0.7);
    } else if (in_kind < 1.5) {
        // Shell: gentle breathing
        float yaw = 0.34 * sin(u_time * 0.72);
        float pitch = 0.12 * cos(u_time * 0.40);
        vec3 rot = rot_x(rot_y(pos, yaw), pitch);
        pos = mix(pos, rot, (1.0 - edge_lock) * 0.42);
        pos.z += 0.042 * sin(u_time * (1.05 + in_speed * 0.18) + in_phase) * (0.65 - 0.4 * edge_lock);
        pos.xy *= 1.0 + pulse * 0.045;
    } else if (in_kind < 2.5) {
        // Fill: expansion + swirl + shockwave
        float yaw = 0.82 * sin(u_time * 0.78);
        float pitch = 0.18 * cos(u_time * 0.45);
        vec3 rot = rot_x(rot_y(pos, yaw), pitch);
        pos = mix(pos, rot, (1.0 - edge_lock) * 0.86);
        float swirl = sin(u_time * (1.45 + in_speed * 0.28) + in_phase);
        pos.x += swirl * 0.05 * core_mix;
        pos.z += cos(u_time * 1.18 + in_phase) * 0.12 * core_mix;
        pos.y += sin(u_time * 1.08 + in_phase * 1.2) * 0.03 * core_mix;
        // Shockwave ripple: radial push
        pos *= 1.0 + shock * 0.025 * (1.0 - abs(in_radial - 0.5) * 2.0);
        pos.xy *= 1.0 + pulse * (0.05 + 0.025 * core_mix);
        pos.z *= 1.0 + pulse * 0.20;
    } else if (in_kind < 3.5) {
        // Core: strong pulse + burst
        pos.xy *= 1.0 + pulse * (0.06 + 0.04 * core_mix);
        pos.z *= 1.0 + pulse * 0.22;
        pos *= 1.0 + shock * 0.04 * core_mix;
    } else {
        // Orbit: figure-8 / circular paths
        float orbit = u_time * (0.95 + in_speed * 0.45) + in_phase;
        float c = cos(orbit), s = sin(orbit);
        float denom = 1.0 + s * s;
        // Blend figure-8 with orbital motion based on phase
        float blend = 0.5 + 0.5 * sin(in_phase * 2.7);
        vec3 fig8 = vec3(
            in_position.x + 0.35 * c / denom,
            in_position.y + 0.35 * s * c / denom * 0.7,
            in_position.z + 0.25 * s / denom
        );
        vec3 circ = vec3(
            in_position.x + 0.12 * cos(orbit),
            in_position.y + 0.08 * sin(orbit * 1.2),
            in_position.z + 0.18 * sin(orbit)
        );
        pos = mix(circ, fig8, blend);
        pos *= 1.0 + pulse * 0.09;
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
    v_color = in_color * depth_tint * depth_light * sil_light * shim_light;

    // HDR brightness for bloom extraction
    v_brightness = 1.0
        + pulse * core_mix * 2.5
        + shock * (1.0 - abs(in_radial - shock) * 3.0) * 1.8
        + core_mix * 0.6;

    // Depth fog
    float fog = exp(-depth * 0.25);
    v_alpha = in_alpha * fog * mix(0.72, 1.18, frontness) * mix(0.9, 1.15, shimmer);
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


# ═══════════════════════════════════════════════════════════════
# Section 5: Camera System
# ═══════════════════════════════════════════════════════════════

class CinematicCamera:
    def __init__(self) -> None:
        self._view = np.eye(4, dtype="f4")
        self._proj = np.eye(4, dtype="f4")

    def update(self, elapsed: float, aspect: float):
        yaw = (0.70 * math.sin(0.12 * elapsed + 1.3)
               + 0.30 * math.sin(0.27 * elapsed + 0.7)
               + 0.12 * math.sin(0.45 * elapsed + 3.9))
        pitch = (0.22 * math.sin(0.15 * elapsed + 2.5)
                 + 0.08 * math.sin(0.38 * elapsed + 0.3)
                 + 0.05 * math.sin(0.52 * elapsed + 1.8))
        radius = 3.50 + 0.55 * math.sin(0.08 * elapsed) + 0.22 * math.sin(0.22 * elapsed + 1.2)
        target_x = 0.10 * math.sin(0.13 * elapsed + 0.6)
        target_y = 0.08 * math.sin(0.17 * elapsed + 1.5)
        target_z = 0.05 * math.sin(0.11 * elapsed + 0.8)

        eye = np.array([
            radius * math.cos(pitch) * math.sin(yaw) + target_x,
            radius * math.sin(pitch) + target_y,
            radius * math.cos(pitch) * math.cos(yaw) + target_z,
        ], dtype="f4")
        target = np.array([target_x, target_y, target_z], dtype="f4")
        up = np.array([0.0, 1.0, 0.0], dtype="f4")
        self._view = self._look_at(eye, target, up)
        self._proj = self._perspective(math.radians(42.0), aspect, 0.1, 20.0)

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

    def _setup_particles(self) -> None:
        noise = BlinnNoise3D(self._args.seed + 1)
        particles = build_particle_cloud(self._args.seed, noise, self._args.particles)
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

        self._camera.update(elapsed, aspect)
        self._heartbeat.update(elapsed)

        view = self._camera.view
        proj = self._camera.proj
        beat = self._heartbeat.beat
        shockwave = self._heartbeat.shockwave
        bloom_intensity = self._heartbeat.bloom_intensity
        point_scale = min(w, h) / 80.0

        # Pass 1: Scene
        if self._enable_bloom:
            self._bloom.begin_scene_pass()
        else:
            self._ctx.screen.use()
            self._ctx.viewport = (0, 0, w, h)
            self._ctx.clear(*BACKGROUND, depth=1.0)

        self._u_proj.write(proj.T.tobytes())
        self._u_view.write(view.T.tobytes())
        self._u_time.value = elapsed
        self._u_beat.value = beat
        self._u_point_scale.value = point_scale
        self._u_shockwave.value = shockwave
        self._particle_vao.render(moderngl.POINTS)

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
        while not glfw.window_should_close(self._window):
            if glfw.get_key(self._window, glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.set_window_should_close(self._window, True)
            if glfw.get_key(self._window, glfw.KEY_SPACE) == glfw.PRESS:
                self._fps_value = 0  # reset on regen hint (not implemented here)

            elapsed = time.perf_counter() - start
            self.render(elapsed)
            glfw.swap_buffers(self._window)
            glfw.poll_events()
            self._frame_count += 1

            if self._show_fps:
                now = time.perf_counter()
                if now - self._fps_timer >= 1.0:
                    self._fps_value = round(self._frame_count / (now - self._fps_timer))
                    self._frame_count = 0
                    self._fps_timer = now
                    glfw.set_window_title(self._window,
                                          f"{WINDOW_TITLE}  |  {self._fps_value} fps  |  {self._particle_count} particles")

        self.close()

    def self_test(self) -> None:
        for i in range(3):
            self.render(i * 0.05)
            glfw.swap_buffers(self._window)
            glfw.poll_events()
        self.close()

    def close(self) -> None:
        try:
            if hasattr(self, "_particle_vao"):
                self._particle_vao.release()
            if hasattr(self, "_vbo"):
                self._vbo.release()
            if hasattr(self, "_prog_particles"):
                self._prog_particles.release()
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
    p.add_argument("--particles", type=int, default=200000,
                   help="Total particle count (default 200000, max ~500000)")
    p.add_argument("--bloom", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable HDR bloom post-processing")
    p.add_argument("--trails", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable motion trail accumulation")
    p.add_argument("--fps", action="store_true", help="Show FPS counter in window title")
    p.add_argument("--self-test", action="store_true",
                   help="Headless render test, exit after a few frames")
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
