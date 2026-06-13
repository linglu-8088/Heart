from __future__ import annotations

import moderngl

from .shaders import QUAD_VS, TRAIL_DECAY_FS


class TrailSystem:
    DECAY = 0.92

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
            vertex_shader=QUAD_VS,
            fragment_shader=TRAIL_DECAY_FS,
        )
        self.quad_decay = ctx.vertex_array(self.prog_decay, [])
        self._read_tex = self.tex_a
        self._write_fbo = self.fbo_b
        self._first_frame = True

    def accumulate(self, scene_tex) -> None:
        if self._first_frame:
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

        self._read_tex.use(location=0)
        self.prog_decay["u_input_tex"] = 0
        self.prog_decay["u_decay"].value = self.DECAY
        self.quad_decay.render(moderngl.TRIANGLES, vertices=6)

        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE
        scene_tex.use(location=0)
        self.prog_decay["u_decay"].value = 0.12
        self.quad_decay.render(moderngl.TRIANGLES, vertices=6)
        self._ctx.blend_func = moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA

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
