from __future__ import annotations

import argparse
import sys
import time

import glfw
import moderngl
import numpy as np

from .bloom import BloomPipeline
from .camera import CinematicCamera
from .constants import (
    BACKGROUND,
    BASE_SPARKS,
    BASE_STARS,
    PEDESTAL_COUNTS,
    THEMES,
    THEME_NAMES,
    WINDOW_TITLE,
)
from .heartbeat import HeartbeatSystem
from .particles import (
    _sample_pedestal,
    _sample_sparks,
    _sample_stars,
    build_particle_cloud,
)
from .shaders import (
    PARTICLE_FS,
    PARTICLE_VS,
    PEDESTAL_FS,
    PEDESTAL_VS,
    SPARK_FS,
    SPARK_VS,
    STAR_FS,
    STAR_VS,
)
from .trails import TrailSystem


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
        self._bloom = None
        self._trail = None
        if self._enable_bloom:
            self._setup_bloom()
        if self._enable_trails:
            self._setup_trail()
        self._camera = CinematicCamera()
        self._heartbeat = HeartbeatSystem()
        self._update_fb_state()
        self._setup_stars()
        self._setup_sparks()
        if self._args.pedestal:
            self._setup_pedestal()

    def _setup_particles(self) -> None:
        particles = build_particle_cloud(self._args.seed, self._args.particles)
        self._particle_count = particles.shape[0]

        self._prog_particles = self._ctx.program(
            vertex_shader=PARTICLE_VS,
            fragment_shader=PARTICLE_FS,
        )
        self._u_proj = self._prog_particles["u_proj"]
        self._u_view = self._prog_particles["u_view"]
        self._u_time = self._prog_particles["u_time"]
        self._u_beat = self._prog_particles["u_beat"]
        self._u_point_scale = self._prog_particles["u_point_scale"]
        self._u_shockwave = self._prog_particles["u_shockwave"]
        self._u_theme_tint = self._prog_particles["u_theme_tint"]
        self._u_light_dir = self._prog_particles["u_light_dir"]

        vbo = self._ctx.buffer(particles.tobytes())
        self._particle_vao = self._ctx.vertex_array(
            self._prog_particles,
            [
                (
                    vbo,
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
        self._vbo = vbo

    def _setup_bloom(self) -> None:
        self._bloom = BloomPipeline(self._ctx, self._args.width, self._args.height)

    def _teardown_bloom(self) -> None:
        if self._bloom is not None:
            self._bloom.release()
            self._bloom = None

    def _setup_trail(self) -> None:
        self._trail = TrailSystem(self._ctx, self._args.width, self._args.height)

    def _teardown_trail(self) -> None:
        if self._trail is not None:
            self._trail.release()
            self._trail = None

    def _setup_stars(self) -> None:
        rng = np.random.default_rng(self._args.seed + 10)
        stars = _sample_stars(rng, BASE_STARS)
        self._prog_stars = self._ctx.program(vertex_shader=STAR_VS, fragment_shader=STAR_FS)
        vbo = self._ctx.buffer(stars.tobytes())
        self._star_vao = self._ctx.vertex_array(
            self._prog_stars,
            [
                (
                    vbo,
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
        self._star_vbo = vbo
        self._star_u_proj = self._prog_stars["u_proj"]
        self._star_u_view = self._prog_stars["u_view"]
        self._star_u_time = self._prog_stars["u_time"]

    def _setup_sparks(self) -> None:
        rng = np.random.default_rng(self._args.seed + 20)
        sparks = _sample_sparks(rng, BASE_SPARKS)
        self._prog_sparks = self._ctx.program(vertex_shader=SPARK_VS, fragment_shader=SPARK_FS)
        vbo = self._ctx.buffer(sparks.tobytes())
        self._spark_vao = self._ctx.vertex_array(
            self._prog_sparks,
            [
                (
                    vbo,
                    "3f 3f 1f 1f 1f 1f",
                    "in_origin",
                    "in_direction",
                    "in_speed",
                    "in_phase_off",
                    "in_particle_size",
                    "in_alpha_val",
                )
            ],
        )
        self._spark_vbo = vbo
        self._spark_u_proj = self._prog_sparks["u_proj"]
        self._spark_u_view = self._prog_sparks["u_view"]
        self._spark_u_time = self._prog_sparks["u_time"]

    def _setup_pedestal(self) -> None:
        rng = np.random.default_rng(self._args.seed + 30)
        count = PEDESTAL_COUNTS.get(self._args.quality, 4200)
        data = _sample_pedestal(rng, count)
        self._prog_pedestal = self._ctx.program(
            vertex_shader=PEDESTAL_VS,
            fragment_shader=PEDESTAL_FS,
        )
        vbo = self._ctx.buffer(data.tobytes())
        self._pedestal_vao = self._ctx.vertex_array(
            self._prog_pedestal,
            [
                (
                    vbo,
                    "3f 1f 1f 1f 1f",
                    "in_position",
                    "in_size",
                    "in_phase",
                    "in_layer",
                    "in_alpha",
                )
            ],
        )
        self._pedestal_vbo = vbo
        self._ped_u_proj = self._prog_pedestal["u_proj"]
        self._ped_u_view = self._prog_pedestal["u_view"]
        self._ped_u_time = self._prog_pedestal["u_time"]
        self._ped_u_beat = self._prog_pedestal["u_beat"]
        self._ped_u_tint = self._prog_pedestal["u_ped_tint"]
        self._ped_u_point_scale = getattr(self._prog_pedestal, "_members", {}).get("u_point_scale")

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
        pedestal_point_scale = point_scale * 0.80

        src = THEMES[self._current_theme]
        dst = THEMES[self._target_theme]
        blend = self._theme_blend
        theme_tints = []
        for index in range(5):
            if blend > 0:
                tint = tuple((1 - blend) * src[index][channel] + blend * dst[index][channel] for channel in range(3))
            else:
                tint = src[index]
            theme_tints.append(tint)
        self._u_theme_tint.value = tuple(theme_tints)
        s = theme_tints
        # Golden crystal pedestal — warm gold tones that complement the red heart
        self._pedestal_tints = [
            tuple(s[2][i] * 0.78 + 0.28 for i in range(3)),
            tuple(s[2][i] * 0.50 + 0.42 for i in range(3)),
            tuple(s[2][i] * 0.40 + 0.22 for i in range(3)),
            tuple(s[3][i] * 0.25 + 0.12 for i in range(3)),
        ]

        if self._enable_bloom:
            self._bloom.begin_scene_pass()
        else:
            self._ctx.screen.use()
            self._ctx.viewport = (0, 0, w, h)
            self._ctx.clear(*BACKGROUND, depth=1.0)

        if hasattr(self, "_star_vao"):
            self._ctx.depth_mask = False
            self._star_u_proj.write(proj.T.tobytes())
            self._star_u_view.write(view.T.tobytes())
            self._star_u_time.value = elapsed
            self._star_vao.render(moderngl.POINTS)
            self._ctx.depth_mask = True

        if hasattr(self, "_pedestal_vao"):
            self._ctx.depth_mask = False
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            self._ped_u_proj.write(proj.T.tobytes())
            self._ped_u_view.write(view.T.tobytes())
            self._ped_u_time.value = elapsed
            self._ped_u_beat.value = beat
            if self._ped_u_point_scale is not None:
                self._ped_u_point_scale.value = pedestal_point_scale
            self._ped_u_tint.value = self._pedestal_tints
            self._pedestal_vao.render(moderngl.POINTS)
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA
            self._ctx.depth_mask = True

        self._u_proj.write(proj.T.tobytes())
        self._u_view.write(view.T.tobytes())
        self._u_time.value = elapsed
        self._u_beat.value = beat
        self._u_point_scale.value = point_scale
        self._u_shockwave.value = shockwave
        self._u_light_dir.value = (0.45, 0.55, 0.65)
        self._particle_vao.render(moderngl.POINTS)

        if hasattr(self, "_spark_vao"):
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
            self._spark_u_proj.write(proj.T.tobytes())
            self._spark_u_view.write(view.T.tobytes())
            self._spark_u_time.value = elapsed
            self._spark_vao.render(moderngl.POINTS)
            self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

        if self._trail is not None and self._enable_bloom:
            self._trail.accumulate(self._bloom.tex_scene)

        trail_tex = self._trail.trail_texture if self._trail is not None else None
        trail_strength = 0.16 if self._trail is not None else 0.0
        if self._enable_bloom:
            self._bloom.execute_post_passes(trail_tex, trail_strength, bloom_intensity)

    def run(self) -> None:
        start = time.perf_counter()
        self._fps_timer = start
        self._setup_glfw_callbacks()
        last_key_state = {}
        _key_repeat_delay = 0.25

        while not glfw.window_should_close(self._window):
            for key_name, key_val in [
                ("ESCAPE", glfw.KEY_ESCAPE),
                ("SPACE", glfw.KEY_SPACE),
                ("C", glfw.KEY_C),
                ("T", glfw.KEY_T),
                ("1", glfw.KEY_1),
                ("2", glfw.KEY_2),
                ("F", glfw.KEY_F),
                ("R", glfw.KEY_R),
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
                        new_mode = (
                            CinematicCamera.MANUAL
                            if self._camera.mode == CinematicCamera.AUTO
                            else CinematicCamera.AUTO
                        )
                        if new_mode == CinematicCamera.MANUAL:
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
                        if self._enable_bloom:
                            if self._bloom is None:
                                self._setup_bloom()
                                w, h = self._fb_size
                                if w > 0 and h > 0:
                                    self._bloom.resize(w, h)
                        else:
                            self._teardown_bloom()
                    elif key_name == "2":
                        self._enable_trails = not self._enable_trails
                        if self._enable_trails:
                            if self._trail is None:
                                self._setup_trail()
                                w, h = self._fb_size
                                if w > 0 and h > 0:
                                    self._trail.resize(w, h)
                        else:
                            self._teardown_trail()
                    elif key_name == "F":
                        self._show_fps = not self._show_fps
                    elif key_name == "R":
                        self._camera.set_manual(0.0, 0.0, 4.0, 0.0, 0.0, 0.0)
                last_key_state[key_name] = pressed

            if self._camera.mode != CinematicCamera.AUTO:
                dx = self._mouse_dx
                dy = self._mouse_dy
                if self._mouse_buttons[0]:
                    yaw = self._camera.manual_yaw - dx * 0.008
                    pitch = self._camera.manual_pitch + dy * 0.008
                    self._camera.set_manual(
                        yaw,
                        pitch,
                        self._camera.manual_radius,
                        self._camera.manual_tx,
                        self._camera.manual_ty,
                        self._camera.manual_tz,
                    )
                if self._mouse_buttons[1]:
                    tx = self._camera.manual_tx + dx * 0.003
                    ty = self._camera.manual_ty - dy * 0.003
                    self._camera.set_manual(
                        self._camera.manual_yaw,
                        self._camera.manual_pitch,
                        self._camera.manual_radius,
                        tx,
                        ty,
                        self._camera.manual_tz,
                    )
                self._mouse_dx = 0.0
                self._mouse_dy = 0.0
            if self._scroll_offset != 0.0:
                radius = self._camera.manual_radius - self._scroll_offset * 0.5
                self._camera.set_manual(
                    self._camera.manual_yaw,
                    self._camera.manual_pitch,
                    radius,
                    self._camera.manual_tx,
                    self._camera.manual_ty,
                    self._camera.manual_tz,
                )
                self._scroll_offset = 0.0

            elapsed = time.perf_counter() - start
            self.render(elapsed)
            glfw.swap_buffers(self._window)
            glfw.poll_events()
            self._frame_count += 1

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
        win = self._window

        def mouse_button_cb(window, button, action, mods):
            if button < 3:
                self._mouse_buttons[button] = action == glfw.PRESS
            if action == glfw.PRESS:
                self._mouse_dx = 0.0
                self._mouse_dy = 0.0
                if self._camera.mode == CinematicCamera.AUTO:
                    self._camera._auto_yaw = 0.0
                    self._camera._auto_pitch = 0.0
                    self._camera._auto_radius = 4.0
                    self._camera._auto_tx = 0.0
                    self._camera._auto_ty = 0.0
                    self._camera._auto_tz = 0.0
                    self._camera._transition_time = 0.0
                    self._camera.mode = CinematicCamera.MANUAL

        def cursor_pos_cb(window, xpos, ypos):
            if any(self._mouse_buttons):
                self._mouse_dx += xpos - self._mouse_x
                self._mouse_dy += ypos - self._mouse_y
            self._mouse_x = xpos
            self._mouse_y = ypos

        def scroll_cb(window, xoff, yoff):
            self._scroll_offset += yoff

        glfw.set_mouse_button_callback(win, mouse_button_cb)
        glfw.set_cursor_pos_callback(win, cursor_pos_cb)
        glfw.set_scroll_callback(win, scroll_cb)

    def self_test(self) -> None:
        for index in range(3):
            self.render(index * 0.05)
            glfw.swap_buffers(self._window)
            glfw.poll_events()
        self.close()

    def close(self) -> None:
        try:
            for attr in [
                "_particle_vao",
                "_vbo",
                "_prog_particles",
                "_star_vao",
                "_star_vbo",
                "_prog_stars",
                "_spark_vao",
                "_spark_vbo",
                "_prog_sparks",
                "_pedestal_vao",
                "_pedestal_vbo",
                "_prog_pedestal",
            ]:
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
