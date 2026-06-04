from __future__ import annotations

import argparse
import math
import sys
import time

import glfw
import moderngl
import numpy as np


WINDOW_TITLE = "3D Particle Heart"
HEART_SCALE = 1.0 / 18.0
BACKGROUND = (0.015, 0.0, 0.025, 1.0)

OUTLINE_COUNT = 9000
SHELL_COUNT = 7000
FILL_COUNT = 24000
CORE_COUNT = 3500
ORBIT_COUNT = 6000


VERTEX_SHADER = """
#version 330

uniform mat4 u_proj;
uniform mat4 u_view;
uniform float u_time;
uniform float u_beat;
uniform float u_point_scale;

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
out float v_kind;
out float v_core_mix;
out float v_seed;
out float v_frontness;

float saturate(float value) {
    return clamp(value, 0.0, 1.0);
}

vec3 rotate_y(vec3 value, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec3(
        c * value.x + s * value.z,
        value.y,
        -s * value.x + c * value.z
    );
}

vec3 rotate_x(vec3 value, float angle) {
    float c = cos(angle);
    float s = sin(angle);
    return vec3(
        value.x,
        c * value.y - s * value.z,
        s * value.y + c * value.z
    );
}

void main() {
    vec3 base = in_position;
    vec3 pos = base;

    float edge_lock = smoothstep(0.78, 1.0, in_radial);
    float core_mix = 1.0 - smoothstep(0.25, 0.95, in_radial);
    float pulse = u_beat;
    float shimmer = 0.5 + 0.5 * sin(u_time * (1.2 + in_speed * 0.35) + in_phase);

    if (in_kind < 0.5) {
        pos.xy *= 1.0 + pulse * 0.055;
        pos.z += 0.015 * sin(u_time * 1.8 + in_phase) * (1.0 - edge_lock);
        pos.z += 0.024 * sin(u_time * 0.75 + in_phase * 0.7);
    } else if (in_kind < 1.5) {
        float yaw = 0.34 * sin(u_time * 0.72);
        float pitch = 0.12 * cos(u_time * 0.40);
        vec3 rotated = rotate_x(rotate_y(base, yaw), pitch);

        pos = mix(base, rotated, (1.0 - edge_lock) * 0.42);
        pos.z += 0.042 * sin(u_time * (1.05 + in_speed * 0.18) + in_phase) * (0.65 - 0.4 * edge_lock);
        pos.xy *= 1.0 + pulse * 0.045;
    } else if (in_kind < 2.5) {
        float yaw = 0.82 * sin(u_time * 0.78);
        float pitch = 0.18 * cos(u_time * 0.45);
        vec3 rotated = rotate_x(rotate_y(base, yaw), pitch);

        // Keep particles near the silhouette anchored to the canonical heart.
        pos = mix(base, rotated, (1.0 - edge_lock) * 0.86);

        float swirl = sin(u_time * (1.45 + in_speed * 0.28) + in_phase);
        pos.x += swirl * 0.05 * core_mix;
        pos.z += cos(u_time * 1.18 + in_phase) * 0.11 * core_mix;
        pos.y += sin(u_time * 1.08 + in_phase * 1.2) * 0.025 * core_mix;

        pos.xy *= 1.0 + pulse * (0.05 + 0.022 * core_mix);
        pos.z *= 1.0 + pulse * 0.20;
    } else if (in_kind < 3.5) {
        float orbit = u_time * (0.95 + in_speed * 0.45) + in_phase;
        pos.x += 0.12 * cos(orbit);
        pos.y += 0.08 * sin(orbit * 1.2);
        pos.z += 0.18 * sin(orbit);
        pos *= 1.0 + pulse * 0.09;
    } else {
        float orbit = u_time * (0.34 + in_speed * 0.58) + in_phase;
        float c = cos(orbit);
        float s = sin(orbit);
        vec3 orbital = vec3(
            base.x * c - base.z * s,
            base.y + 0.11 * sin(orbit * 1.4 + in_phase),
            base.x * s + base.z * c
        );
        pos = rotate_x(orbital, 0.56 + 0.08 * sin(in_phase));
        pos.xy *= 1.0 + pulse * 0.02;
    }

    float drift_yaw = 0.10 * sin(u_time * 0.34);
    float drift_pitch = 0.06 * cos(u_time * 0.28);
    if (in_kind < 1.5) {
        drift_yaw *= 0.28;
        drift_pitch *= 0.24;
    }
    pos = rotate_x(rotate_y(pos, drift_yaw), drift_pitch);

    vec4 view_pos = u_view * vec4(pos, 1.0);
    gl_Position = u_proj * view_pos;

    float frontness = saturate(pos.z * 1.15 + 0.5);
    float depth_scale = 1.55 / max(2.0, -view_pos.z);
    float pulse_size = 1.0 + pulse * mix(0.09, 0.22, core_mix);
    float front_size = mix(0.84, 1.22, frontness);
    gl_PointSize = clamp(in_size * u_point_scale * depth_scale * pulse_size * front_size, 1.2, 30.0);

    float depth_light = saturate(1.28 - 0.16 * abs(view_pos.z));
    float silhouette_light = mix(0.9, 0.96, edge_lock);
    float shimmer_light = mix(0.95, 1.04, shimmer);
    vec3 depth_tint = mix(
        vec3(0.72, 0.78, 0.96),
        vec3(1.10, 0.98, 0.92),
        frontness
    );

    if (in_kind > 3.5) {
        silhouette_light *= 1.03;
    }

    v_color = in_color * depth_tint * depth_light * silhouette_light * shimmer_light;
    v_alpha = in_alpha * mix(0.72, 1.18, frontness) * mix(0.9, 1.15, shimmer);
    v_kind = in_kind;
    v_core_mix = core_mix;
    v_seed = in_phase;
    v_frontness = frontness;
}
"""


FRAGMENT_SHADER = """
#version 330

in vec3 v_color;
in float v_alpha;
in float v_kind;
in float v_core_mix;
in float v_seed;
in float v_frontness;

out vec4 fragColor;

void main() {
    vec2 uv = gl_PointCoord * 2.0 - 1.0;
    float r2 = dot(uv, uv);
    if (r2 > 1.0) {
        discard;
    }

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
        glow = mix(0.95, 1.06, halo);
    } else {
        alpha_shape *= mix(0.82, 1.0, soft);
        glow = mix(0.94, 1.08, halo);
    }

    glow *= mix(0.92, 1.08, v_frontness);
    glow *= mix(1.0, 1.05, v_core_mix);

    fragColor = vec4(v_color * glow, v_alpha * alpha_shape);
}
"""


def heart_boundary(theta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = 16.0 * np.sin(theta) ** 3
    y = (
        13.0 * np.cos(theta)
        - 5.0 * np.cos(2.0 * theta)
        - 2.0 * np.cos(3.0 * theta)
        - np.cos(4.0 * theta)
    )
    return x * HEART_SCALE, y * HEART_SCALE


def normalize(vector: np.ndarray) -> np.ndarray:
    length = np.linalg.norm(vector)
    if length == 0.0:
        return vector
    return vector / length


def perspective(fov_y: float, aspect: float, near: float, far: float) -> np.ndarray:
    f = 1.0 / math.tan(fov_y / 2.0)
    matrix = np.zeros((4, 4), dtype="f4")
    matrix[0, 0] = f / aspect
    matrix[1, 1] = f
    matrix[2, 2] = (far + near) / (near - far)
    matrix[2, 3] = (2.0 * far * near) / (near - far)
    matrix[3, 2] = -1.0
    return matrix


def look_at(eye: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    forward = normalize(target - eye)
    side = normalize(np.cross(forward, up))
    up_corrected = np.cross(side, forward)

    matrix = np.eye(4, dtype="f4")
    matrix[0, :3] = side
    matrix[1, :3] = up_corrected
    matrix[2, :3] = -forward
    matrix[:3, 3] = -matrix[:3, :3] @ eye
    return matrix


def lerp_colors(color_a: np.ndarray, color_b: np.ndarray, amount: np.ndarray) -> np.ndarray:
    amount = amount[:, None]
    return color_a + (color_b - color_a) * amount


def stack_particles(
    positions: np.ndarray,
    colors: np.ndarray,
    sizes: np.ndarray,
    phase: np.ndarray,
    kind: float,
    radial: np.ndarray,
    alpha: np.ndarray,
    speed: np.ndarray,
) -> np.ndarray:
    kind_column = np.full((positions.shape[0], 1), kind, dtype="f4")
    return np.hstack(
        [
            positions.astype("f4"),
            colors.astype("f4"),
            sizes[:, None].astype("f4"),
            phase[:, None].astype("f4"),
            kind_column,
            radial[:, None].astype("f4"),
            alpha[:, None].astype("f4"),
            speed[:, None].astype("f4"),
        ]
    )


def sample_outline(rng: np.random.Generator, count: int) -> np.ndarray:
    theta = np.linspace(0.0, math.tau, count, endpoint=False)
    theta += rng.normal(0.0, 0.012, count)
    base_x, base_y = heart_boundary(theta)
    shell = 0.95 + 0.08 * rng.random(count) + rng.normal(0.0, 0.012, count)
    jitter = rng.normal(0.0, 0.014, (count, 2))
    x = base_x * shell + jitter[:, 0]
    y = base_y * shell + jitter[:, 1]
    z = rng.normal(0.0, 0.09, count)

    positions = np.column_stack([x, y, z])
    color_mix = rng.random(count)
    colors = lerp_colors(
        np.array([0.94, 0.14, 0.28], dtype="f4"),
        np.array([0.98, 0.30, 0.46], dtype="f4"),
        color_mix,
    )
    sizes = rng.uniform(2.9, 5.0, count)
    phase = rng.uniform(0.0, math.tau, count)
    radial = np.clip(shell, 0.82, 1.04)
    alpha = rng.uniform(0.16, 0.30, count)
    speed = rng.uniform(0.8, 1.3, count)
    return stack_particles(positions, colors, sizes, phase, 0.0, radial, alpha, speed)


def sample_shell(rng: np.random.Generator, count: int) -> np.ndarray:
    theta = rng.uniform(0.0, math.tau, count)
    base_x, base_y = heart_boundary(theta)
    radial = 0.62 + 0.34 * (rng.random(count) ** 0.72)
    shell_noise = rng.normal(0.0, 0.018, (count, 2))

    x = base_x * radial + shell_noise[:, 0]
    y = base_y * radial + shell_noise[:, 1]
    thickness = 0.07 + 0.18 * (1.0 - radial)
    z = rng.normal(0.0, thickness, count)
    z = np.clip(z, -0.40, 0.40)

    positions = np.column_stack([x, y, z])
    color_mix = np.clip(0.35 + 0.55 * rng.random(count), 0.0, 1.0)
    colors = lerp_colors(
        np.array([0.96, 0.20, 0.32], dtype="f4"),
        np.array([0.98, 0.42, 0.52], dtype="f4"),
        color_mix,
    )
    sizes = rng.uniform(2.5, 4.8, count)
    phase = rng.uniform(0.0, math.tau, count)
    alpha = rng.uniform(0.08, 0.16, count)
    speed = rng.uniform(0.9, 1.6, count)
    return stack_particles(positions, colors, sizes, phase, 1.0, radial, alpha, speed)


def sample_fill(rng: np.random.Generator, count: int) -> np.ndarray:
    theta = rng.uniform(0.0, math.tau, count)
    base_x, base_y = heart_boundary(theta)
    radial = 0.14 + 0.86 * np.sqrt(rng.random(count))
    noise_scale = 0.015 + 0.02 * (1.0 - radial)

    x = base_x * radial + rng.normal(0.0, noise_scale, count)
    y = base_y * radial + rng.normal(0.0, noise_scale, count)
    thickness = 0.06 + 0.44 * (1.0 - radial**1.45)
    z = rng.normal(0.0, thickness, count)
    z = np.clip(z, -0.62, 0.62)

    positions = np.column_stack([x, y, z])
    warm_mix = np.clip(radial * 0.9 + rng.normal(0.0, 0.08, count), 0.0, 1.0)
    colors = lerp_colors(
        np.array([0.98, 0.46, 0.38], dtype="f4"),
        np.array([0.94, 0.12, 0.28], dtype="f4"),
        warm_mix,
    )
    colors = lerp_colors(colors, np.array([0.98, 0.54, 0.48], dtype="f4"), (1.0 - radial) * 0.12)
    sizes = rng.uniform(1.8, 3.8, count)
    phase = rng.uniform(0.0, math.tau, count)
    alpha = rng.uniform(0.08, 0.14, count)
    speed = rng.uniform(0.7, 1.5, count)
    return stack_particles(positions, colors, sizes, phase, 2.0, radial, alpha, speed)


def sample_core(rng: np.random.Generator, count: int) -> np.ndarray:
    theta = rng.uniform(0.0, math.tau, count)
    radius = rng.random(count) ** 1.9 * 0.24
    x = radius * np.cos(theta) * 0.95 + rng.normal(0.0, 0.03, count)
    y = radius * np.sin(theta) * 1.05 + rng.normal(0.0, 0.03, count)
    z = rng.normal(0.0, 0.18, count)

    positions = np.column_stack([x, y, z])
    color_mix = rng.random(count)
    colors = lerp_colors(
        np.array([0.98, 0.40, 0.52], dtype="f4"),
        np.array([0.99, 0.56, 0.64], dtype="f4"),
        color_mix,
    )
    sizes = rng.uniform(3.8, 6.8, count)
    phase = rng.uniform(0.0, math.tau, count)
    radial = np.clip(radius / 0.24, 0.0, 0.35)
    alpha = rng.uniform(0.05, 0.10, count)
    speed = rng.uniform(1.0, 2.1, count)
    return stack_particles(positions, colors, sizes, phase, 3.0, radial, alpha, speed)


def sample_orbit(rng: np.random.Generator, count: int) -> np.ndarray:
    angle = rng.uniform(0.0, math.tau, count)
    radius = rng.uniform(1.05, 1.95, count)
    x = radius * np.cos(angle)
    z = radius * 0.82 * np.sin(angle)
    y = rng.normal(0.0, 0.42, count) + 0.22 * np.sin(angle * 2.4)

    positions = np.column_stack([x, y, z])
    color_mix = np.clip(rng.normal(0.48, 0.22, count), 0.0, 1.0)
    colors = lerp_colors(
        np.array([1.0, 0.25, 0.22], dtype="f4"),
        np.array([0.98, 0.58, 0.42], dtype="f4"),
        color_mix,
    )
    sizes = rng.uniform(1.8, 4.4, count)
    phase = rng.uniform(0.0, math.tau, count)
    radial = np.full(count, 0.18, dtype="f4")
    alpha = rng.uniform(0.04, 0.08, count)
    speed = rng.uniform(0.8, 1.9, count)
    return stack_particles(positions, colors, sizes, phase, 4.0, radial, alpha, speed)


def build_particle_cloud(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    clouds = [
        sample_outline(rng, OUTLINE_COUNT),
        sample_shell(rng, SHELL_COUNT),
        sample_fill(rng, FILL_COUNT),
        sample_core(rng, CORE_COUNT),
        sample_orbit(rng, ORBIT_COUNT),
    ]
    particles = np.vstack(clouds).astype("f4")
    rng.shuffle(particles, axis=0)
    return particles


def heartbeat_value(seconds: float) -> float:
    beat_time = (seconds * 1.18) % 1.0
    beat_a = math.exp(-((beat_time - 0.12) / 0.05) ** 2)
    beat_b = 0.42 * math.exp(-((beat_time - 0.30) / 0.04) ** 2)
    return beat_a + beat_b


class ParticleHeartApp:
    def __init__(self, width: int, height: int, self_test: bool, seed: int) -> None:
        if not glfw.init():
            raise RuntimeError("GLFW initialization failed.")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.DEPTH_BITS, 24)
        if sys.platform == "darwin":
            glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, glfw.TRUE)
        if self_test:
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)

        self.window = glfw.create_window(width, height, WINDOW_TITLE, None, None)
        if not self.window:
            glfw.terminate()
            raise RuntimeError("Failed to create an OpenGL window.")

        glfw.make_context_current(self.window)
        glfw.swap_interval(1 if not self_test else 0)

        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.BLEND | moderngl.PROGRAM_POINT_SIZE | moderngl.DEPTH_TEST)
        self.ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        self.prog = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=FRAGMENT_SHADER)
        self.u_proj = self.prog["u_proj"]
        self.u_view = self.prog["u_view"]
        self.u_time = self.prog["u_time"]
        self.u_beat = self.prog["u_beat"]
        self.u_point_scale = self.prog["u_point_scale"]

        particles = build_particle_cloud(seed)
        self.vbo = self.ctx.buffer(particles.tobytes())
        self.vao = self.ctx.vertex_array(
            self.prog,
            [
                (
                    self.vbo,
                    "3f 3f 1f 1f 1f 1f 1f 1f",
                    "in_position",
                    "in_color",
                    "in_size",
                    "in_phase",
                    "in_kind",
                    "in_radial",
                    "in_alpha",
                    "in_speed",
                )
            ],
        )

        self.framebuffer_size = (-1, -1)
        self.update_framebuffer_state()

    def update_framebuffer_state(self) -> None:
        width, height = glfw.get_framebuffer_size(self.window)
        width = max(width, 1)
        height = max(height, 1)
        if (width, height) == self.framebuffer_size:
            return

        self.framebuffer_size = (width, height)
        self.ctx.viewport = (0, 0, width, height)

        aspect = width / height
        proj = perspective(math.radians(42.0), aspect, 0.1, 20.0)
        self.u_proj.write(proj.T.tobytes())

    def update_camera(self, elapsed: float) -> None:
        eye = np.array(
            [
                0.62 * math.sin(elapsed * 0.54),
                0.28 * math.cos(elapsed * 0.37),
                3.72 + 0.20 * math.sin(elapsed * 0.24),
            ],
            dtype="f4",
        )
        target = np.array(
            [
                0.06 * math.sin(elapsed * 0.33),
                0.04 * math.sin(elapsed * 0.55),
                0.0,
            ],
            dtype="f4",
        )
        up = np.array([0.0, 1.0, 0.0], dtype="f4")
        view = look_at(eye, target, up)
        self.u_view.write(view.T.tobytes())

    def render(self, elapsed: float) -> None:
        self.update_framebuffer_state()
        self.update_camera(elapsed)
        self.ctx.clear(*BACKGROUND, depth=1.0)

        heartbeat = heartbeat_value(elapsed)
        point_scale = min(self.framebuffer_size) / 72.0

        self.u_time.value = elapsed
        self.u_beat.value = heartbeat
        self.u_point_scale.value = point_scale
        self.vao.render(mode=moderngl.POINTS)

    def run(self) -> None:
        start = time.perf_counter()
        while not glfw.window_should_close(self.window):
            if glfw.get_key(self.window, glfw.KEY_ESCAPE) == glfw.PRESS:
                glfw.set_window_should_close(self.window, True)

            elapsed = time.perf_counter() - start
            self.render(elapsed)
            glfw.swap_buffers(self.window)
            glfw.poll_events()

        self.close()

    def self_test(self) -> None:
        for index in range(3):
            self.render(index * 0.05)
            glfw.swap_buffers(self.window)
            glfw.poll_events()
        self.close()

    def close(self) -> None:
        try:
            self.vao.release()
            self.vbo.release()
            self.prog.release()
            self.ctx.release()
        finally:
            if self.window is not None:
                glfw.destroy_window(self.window)
            glfw.terminate()
            self.window = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a 3D particle heart with a stable silhouette.")
    parser.add_argument("--width", type=int, default=1280, help="Window width in pixels.")
    parser.add_argument("--height", type=int, default=960, help="Window height in pixels.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for deterministic particles.")
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Create a hidden OpenGL context, render a few frames, and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = ParticleHeartApp(args.width, args.height, args.self_test, args.seed)
    if args.self_test:
        app.self_test()
        print("Self-test passed.")
    else:
        app.run()


if __name__ == "__main__":
    main()
