from __future__ import annotations

import math

import numpy as np


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

        self.manual_yaw = 0.0
        self.manual_pitch = 0.0
        self.manual_radius = 4.0
        self.manual_tx = 0.0
        self.manual_ty = 0.0
        self.manual_tz = 0.0

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

    def set_manual(
        self,
        yaw: float,
        pitch: float,
        radius: float,
        tx: float = 0.0,
        ty: float = 0.0,
        tz: float = 0.0,
    ) -> None:
        self.manual_yaw = yaw
        self.manual_pitch = max(-1.3, min(1.3, pitch))
        self.manual_radius = max(1.5, min(15.0, radius))
        self.manual_tx = tx
        self.manual_ty = ty
        self.manual_tz = tz

    def reset(self, aspect: float) -> None:
        self.set_manual(0.0, 0.0, 4.0)
        self._rebuild(aspect, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0)

    def _rebuild(self, aspect, yaw, pitch, radius, tx, ty, tz):
        eye = np.array(
            [
                radius * math.cos(pitch) * math.sin(yaw) + tx,
                radius * math.sin(pitch) + ty,
                radius * math.cos(pitch) * math.cos(yaw) + tz,
            ],
            dtype="f4",
        )
        target = np.array([tx, ty, tz], dtype="f4")
        up = np.array([0.0, 1.0, 0.0], dtype="f4")
        self._view = self._look_at(eye, target, up)
        self._proj = self._perspective(math.radians(38.0), aspect, 0.1, 20.0)

    def update(self, elapsed: float, aspect: float):
        a_yaw = 0.0
        a_pitch = 0.0
        a_radius = (
            3.55
            + 0.12 * math.sin(elapsed * 0.18 + 0.5)
            + 0.06 * math.sin(elapsed * 0.31 + 2.1)
        )
        a_tx = 0.0
        a_ty = 0.010 * math.sin(elapsed * 0.55)
        a_tz = 0.0

        if self.mode == self.AUTO:
            self._rebuild(aspect, a_yaw, a_pitch, a_radius, a_tx, a_ty, a_tz)
        elif self.mode == self.MANUAL:
            self._rebuild(
                aspect,
                self.manual_yaw,
                self.manual_pitch,
                self.manual_radius,
                self.manual_tx,
                self.manual_ty,
                self.manual_tz,
            )
        elif self.mode == self.TRANSITION:
            t = min(1.0, (elapsed - self._transition_time) / self._transition_duration)
            t_smooth = t * t * (3.0 - 2.0 * t)
            if self._transition_target_mode == self.MANUAL:
                yaw = self._auto_yaw + (self.manual_yaw - self._auto_yaw) * t_smooth
                pitch = self._auto_pitch + (self.manual_pitch - self._auto_pitch) * t_smooth
                radius = self._auto_radius + (self.manual_radius - self._auto_radius) * t_smooth
                tx = self._auto_tx + (self.manual_tx - self._auto_tx) * t_smooth
                ty = self._auto_ty + (self.manual_ty - self._auto_ty) * t_smooth
                tz = self._auto_tz + (self.manual_tz - self._auto_tz) * t_smooth
            else:
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
