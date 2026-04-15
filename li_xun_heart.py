import colorsys
import math
import random
import tkinter as tk
from dataclasses import dataclass


@dataclass
class Particle:
    home_x: float
    home_y: float
    x: float
    y: float
    base_size: float
    color: str
    kind: str
    depth: float
    wobble: float
    speed: float
    phase: float
    twinkle: float
    item_id: int = 0
    vx: float = 0.0
    vy: float = 0.0
    size: float = 1.0


class HeartAnimation:
    """Recommended main version for the project."""

    def __init__(self, width: int = 900, height: int = 680) -> None:
        self.width = width
        self.height = height
        self.center_x = width // 2
        self.center_y = height // 2 - 20
        self.frame = 0
        self.time = 0.0
        self.heartbeat = 0.0
        self.running = True

        self.root = tk.Tk()
        self.root.title("Li Xun Heart")
        self.root.geometry(f"{width}x{height}")
        self.root.configure(bg="#05070d")
        self.root.resizable(False, False)

        self.canvas = tk.Canvas(
            self.root,
            width=width,
            height=height,
            bg="#05070d",
            highlightthickness=0,
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.heart_particles: list[Particle] = []
        self.ground_particles: list[Particle] = []
        self.spark_particles: list[Particle] = []
        self.star_items: list[int] = []

        self._build_scene()
        self._bind_events()

    def _bind_events(self) -> None:
        self.root.bind("<Escape>", lambda _event: self.root.destroy())
        self.root.bind("<space>", lambda _event: self.reseed_scene())
        self.root.bind("r", lambda _event: self.reseed_scene())
        self.root.bind("<Button-1>", lambda _event: self.reseed_scene())

    @staticmethod
    def hsv_to_hex(h: float, s: float, v: float) -> str:
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        return f"#{int(r * 255):02x}{int(g * 255):02x}{int(b * 255):02x}"

    def heart_point(self, t: float, scale: float = 1.0) -> tuple[float, float]:
        x = 16 * math.sin(t) ** 3
        y = -(13 * math.cos(t) - 5 * math.cos(2 * t) - 2 * math.cos(3 * t) - math.cos(4 * t))
        return x * scale, y * scale

    def make_heart_particle(self, scale: float, kind: str) -> Particle:
        t = random.uniform(0, math.tau)
        px, py = self.heart_point(t, scale=scale)
        x = self.center_x + px * 11.5
        y = self.center_y + py * 11.5

        if kind == "glow":
            base_size = random.uniform(2.6, 4.8)
            color = self.hsv_to_hex(
                random.uniform(0.95, 0.99),
                random.uniform(0.55, 0.8),
                random.uniform(0.22, 0.35),
            )
            depth = random.uniform(0.45, 0.7)
            wobble = random.uniform(1.6, 3.4)
        elif kind == "core":
            base_size = random.uniform(1.5, 2.8)
            color = self.hsv_to_hex(
                random.uniform(0.96, 1.0),
                random.uniform(0.55, 0.85),
                random.uniform(0.92, 1.0),
            )
            depth = random.uniform(0.9, 1.15)
            wobble = random.uniform(0.4, 1.0)
        else:
            base_size = random.uniform(0.9, 2.1)
            color = self.hsv_to_hex(
                random.uniform(0.94, 0.99),
                random.uniform(0.65, 0.95),
                random.uniform(0.6, 0.95),
            )
            depth = random.uniform(0.75, 1.0)
            wobble = random.uniform(0.6, 1.6)

        particle = Particle(
            home_x=x,
            home_y=y,
            x=x,
            y=y,
            base_size=base_size,
            color=color,
            kind=kind,
            depth=depth,
            wobble=wobble,
            speed=random.uniform(0.7, 1.35),
            phase=random.uniform(0, math.tau),
            twinkle=random.uniform(0.8, 1.5),
        )
        particle.size = base_size
        return particle

    def make_ground_particle(self) -> Particle:
        angle = random.uniform(0, math.tau)
        radius_x = random.uniform(120, 255)
        radius_y = random.uniform(10, 28)
        x = self.center_x + math.cos(angle) * radius_x
        y = self.center_y + 235 + math.sin(angle) * radius_y
        color = self.hsv_to_hex(
            random.uniform(0.52, 0.6),
            random.uniform(0.45, 0.82),
            random.uniform(0.45, 0.95),
        )
        particle = Particle(
            home_x=x,
            home_y=y,
            x=x,
            y=y,
            base_size=random.uniform(1.2, 2.6),
            color=color,
            kind="ground",
            depth=random.uniform(0.4, 0.85),
            wobble=random.uniform(4.0, 10.0),
            speed=random.uniform(0.18, 0.5),
            phase=random.uniform(0, math.tau),
            twinkle=random.uniform(0.25, 0.75),
        )
        particle.size = particle.base_size
        return particle

    def make_spark_particle(self) -> Particle:
        t = random.uniform(0, math.tau)
        px, py = self.heart_point(t, scale=random.uniform(1.02, 1.18))
        x = self.center_x + px * 12.4
        y = self.center_y + py * 12.4
        color = self.hsv_to_hex(random.uniform(0.1, 0.16), random.uniform(0.2, 0.5), 1.0)
        particle = Particle(
            home_x=x,
            home_y=y,
            x=x,
            y=y,
            base_size=random.uniform(0.9, 2.2),
            color=color,
            kind="spark",
            depth=random.uniform(0.8, 1.3),
            wobble=random.uniform(3.5, 7.0),
            speed=random.uniform(1.0, 2.2),
            phase=random.uniform(0, math.tau),
            twinkle=random.uniform(1.6, 3.4),
        )
        particle.size = particle.base_size
        return particle

    def _draw_background(self) -> None:
        self.canvas.delete("background")
        palette = ["#05070d", "#071019", "#0a1524", "#09121d", "#070d16", "#05070d"]
        band_height = self.height / len(palette)
        for index, color in enumerate(palette):
            y0 = band_height * index
            y1 = band_height * (index + 1)
            self.canvas.create_rectangle(0, y0, self.width, y1, fill=color, outline="", tags="background")

        for color, bounds in (
            ("#130813", (100, 90, 330, 290)),
            ("#0c1a29", (self.width - 340, 120, self.width - 90, 320)),
            ("#1b0a17", (210, 250, 520, 520)),
            ("#081424", (self.width - 470, 270, self.width - 120, 560)),
        ):
            self.canvas.create_oval(*bounds, fill=color, outline="", tags="background")

        self.star_items.clear()
        for _ in range(70):
            x = random.uniform(20, self.width - 20)
            y = random.uniform(20, self.height - 250)
            size = random.choice((1.0, 1.0, 1.2, 1.5))
            color = random.choice(("#dce9ff", "#ffeef5", "#c9e7ff"))
            item_id = self.canvas.create_oval(
                x - size,
                y - size,
                x + size,
                y + size,
                fill=color,
                outline="",
                tags="background",
            )
            self.star_items.append(item_id)

    def _create_canvas_items(self) -> None:
        for particle in self.ground_particles:
            particle.item_id = self.canvas.create_oval(0, 0, 0, 0, fill=particle.color, outline="")

        for particle in self.heart_particles:
            particle.item_id = self.canvas.create_oval(0, 0, 0, 0, fill=particle.color, outline="")

        for particle in self.spark_particles:
            particle.item_id = self.canvas.create_oval(0, 0, 0, 0, fill=particle.color, outline="")

        self.tip_glow = self.canvas.create_oval(0, 0, 0, 0, fill="#300513", outline="")
        self.core_glow = self.canvas.create_oval(0, 0, 0, 0, fill="#19040d", outline="")
        self.highlight = self.canvas.create_oval(0, 0, 0, 0, fill="#ffe2ec", outline="")
        self.canvas.tag_lower(self.tip_glow)
        self.canvas.tag_raise(self.highlight)

    def _build_scene(self) -> None:
        self.canvas.delete("all")
        self._draw_background()

        self.heart_particles = []
        self.ground_particles = []
        self.spark_particles = []

        for _ in range(360):
            self.heart_particles.append(self.make_heart_particle(random.uniform(1.02, 1.16), "glow"))
        for _ in range(950):
            scale = 0.18 + (random.random() ** 0.58) * 0.82
            self.heart_particles.append(self.make_heart_particle(scale, "fill"))
        for _ in range(420):
            self.heart_particles.append(self.make_heart_particle(random.uniform(0.94, 1.03), "core"))
        for _ in range(280):
            self.ground_particles.append(self.make_ground_particle())
        for _ in range(115):
            self.spark_particles.append(self.make_spark_particle())

        self._create_canvas_items()
        self.update_frame(initial=True)

    def reseed_scene(self) -> None:
        self.frame = 0
        self.time = random.uniform(0.0, 12.0)
        self._build_scene()

    def heartbeat_scale(self) -> float:
        beat = max(0.0, math.sin(self.time * 1.9)) ** 6
        echo = 0.45 * max(0.0, math.sin(self.time * 1.9 - 0.55)) ** 10
        breath = 0.012 * math.sin(self.time * 0.5)
        self.heartbeat = beat + echo
        return 1.0 + breath + 0.075 * self.heartbeat

    def update_glow_layers(self, scale: float) -> None:
        glow_radius_x = 235 * scale
        glow_radius_y = 210 * scale
        self.canvas.coords(
            self.tip_glow,
            self.center_x - glow_radius_x,
            self.center_y - glow_radius_y + 26,
            self.center_x + glow_radius_x,
            self.center_y + glow_radius_y + 26,
        )
        self.canvas.coords(
            self.core_glow,
            self.center_x - 185 * scale,
            self.center_y - 175 * scale,
            self.center_x + 185 * scale,
            self.center_y + 175 * scale,
        )
        shimmer = 4.5 + 3.5 * self.heartbeat
        self.canvas.coords(
            self.highlight,
            self.center_x - 22 - shimmer,
            self.center_y - 155 - shimmer,
            self.center_x + 22 + shimmer,
            self.center_y - 111 + shimmer,
        )

    def update_frame(self, initial: bool = False) -> None:
        self.time += 0.045
        self.frame += 1
        scale = self.heartbeat_scale()
        self.update_glow_layers(scale)

        for index, item_id in enumerate(self.star_items):
            if index % 5 == self.frame % 5:
                twinkle = 0.72 + 0.28 * math.sin(self.time * 1.8 + index)
                self.canvas.itemconfigure(item_id, state="normal" if twinkle >= 0.75 else "hidden")

        for particle in self.ground_particles:
            drift_x = math.sin(self.time * particle.speed + particle.phase) * particle.wobble
            drift_y = math.cos(self.time * particle.speed * 0.7 + particle.phase) * (particle.wobble * 0.15)
            pulse = 1.0 + 0.16 * self.heartbeat * particle.depth
            particle.size = particle.base_size * pulse
            particle.x = particle.home_x + drift_x
            particle.y = particle.home_y + drift_y
            size = particle.size
            self.canvas.coords(
                particle.item_id,
                particle.x - size,
                particle.y - size,
                particle.x + size,
                particle.y + size,
            )

        for particle in self.heart_particles:
            dx = particle.home_x - self.center_x
            dy = particle.home_y - self.center_y
            target_x = self.center_x + dx * scale
            target_y = self.center_y + dy * scale

            flutter_x = math.cos(self.time * particle.speed + particle.phase) * particle.wobble
            flutter_y = math.sin(self.time * (particle.speed * 1.15) + particle.phase) * particle.wobble
            lift = -1.5 * self.heartbeat * particle.depth

            particle.vx = (target_x + flutter_x - particle.x) * 0.18 + particle.vx * 0.73
            particle.vy = (target_y + flutter_y + lift - particle.y) * 0.18 + particle.vy * 0.73
            particle.x += particle.vx
            particle.y += particle.vy

            size_pulse = 1.0 + 0.16 * self.heartbeat * particle.depth
            twinkle = 0.9 + 0.1 * math.sin(self.time * particle.twinkle + particle.phase)
            particle.size = particle.base_size * size_pulse * twinkle
            size = particle.size

            self.canvas.coords(
                particle.item_id,
                particle.x - size,
                particle.y - size,
                particle.x + size,
                particle.y + size,
            )

        for particle in self.spark_particles:
            angle = self.time * particle.speed + particle.phase
            orbit = 10 + particle.wobble + 12 * self.heartbeat
            particle.x = particle.home_x + math.cos(angle) * orbit
            particle.y = particle.home_y + math.sin(angle * 1.2) * (orbit * 0.55) - 14 * self.heartbeat
            particle.size = particle.base_size * (
                0.85 + 0.45 * abs(math.sin(self.time * particle.twinkle + particle.phase))
            )
            size = particle.size
            self.canvas.coords(
                particle.item_id,
                particle.x - size,
                particle.y - size,
                particle.x + size,
                particle.y + size,
            )

        if not initial and self.running:
            self.root.after(16, self.update_frame)

    def run(self) -> None:
        self.update_frame()
        self.root.mainloop()


if __name__ == "__main__":
    HeartAnimation().run()

