from __future__ import annotations

import moderngl

from .constants import BACKGROUND
from .shaders import BLUR_H_FS, BLUR_V_FS, BRIGHT_PASS_FS, COMPOSITE_FS, QUAD_VS


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
            vertex_shader=QUAD_VS,
            fragment_shader=BRIGHT_PASS_FS,
        )
        self.prog_blur_h = self._ctx.program(
            vertex_shader=QUAD_VS,
            fragment_shader=BLUR_H_FS,
        )
        self.prog_blur_v = self._ctx.program(
            vertex_shader=QUAD_VS,
            fragment_shader=BLUR_V_FS,
        )
        self.prog_composite = self._ctx.program(
            vertex_shader=QUAD_VS,
            fragment_shader=COMPOSITE_FS,
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

        self.fbo_bright_a.use()
        ctx.viewport = (0, 0, hw, hh)
        self.fbo_bright_a.clear(0.0, 0.0, 0.0, 1.0)
        self.tex_scene.use(location=0)
        self.prog_bright["u_threshold"].value = 0.72
        self.prog_bright["u_intensity"].value = bloom_intensity
        self._quad_bright.render(moderngl.TRIANGLES, vertices=6)

        self.fbo_bright_b.use()
        self.fbo_bright_b.clear(0.0, 0.0, 0.0, 1.0)
        self.tex_bright_a.use(location=0)
        self.prog_blur_h["u_texel_size"].value = (1.0 / hw, 0.0)
        self._quad_blur_h.render(moderngl.TRIANGLES, vertices=6)

        self.fbo_bright_a.use()
        self.fbo_bright_a.clear(0.0, 0.0, 0.0, 1.0)
        self.tex_bright_b.use(location=0)
        self.prog_blur_v["u_texel_size"].value = (0.0, 1.0 / hh)
        self._quad_blur_v.render(moderngl.TRIANGLES, vertices=6)

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
        for attr in [
            "tex_scene",
            "depth_rbo",
            "fbo_scene",
            "tex_bright_a",
            "fbo_bright_a",
            "tex_bright_b",
            "fbo_bright_b",
        ]:
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()
        for attr in ["prog_bright", "prog_blur_h", "prog_blur_v", "prog_composite"]:
            obj = getattr(self, attr, None)
            if obj is not None:
                obj.release()
