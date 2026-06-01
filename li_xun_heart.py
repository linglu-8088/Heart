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
        self.highlight_items: list[int] = []

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

    @staticmethod
    def heart_contains(x: float, y: float) -> bool:
        flipped_y = -y
        return (x * x + flipped_y * flipped_y - 1.0) ** 3 - x * x * flipped_y ** 3 <= 0.0

    def heart_to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return self.center_x + x * 168.0, self.center_y + 22.0 + y * 146.0

    def sample_heart_xy(self, kind: str) -> tuple[float, float]:
        if kind == "glow":
            t = random.uniform(0, math.tau)
            px, py = self.heart_point(t, scale=random.uniform(0.98, 1.08))
            return (
                self.center_x + px * 10.3 + random.gauss(0.0, 2.8),
                self.center_y + 12.0 + py * 10.3 + random.gauss(0.0, 2.8),
            )

        for _ in range(120):
            x = random.uniform(-1.15, 1.15)
            y = random.uniform(-1.08, 1.18)
            if not self.heart_contains(x, y):
                continue

            if kind == "core":
                if abs(x) < 0.15 and y < 0.7 and random.random() < 0.88:
                    continue
                if y > 0.86 and random.random() < 0.55:
                    continue
                if y < 0.08 and 0.22 < abs(x) < 0.72 and random.random() < 0.4:
                    x *= 1.05
            else:
                if abs(x) < 0.08 and y < 0.76 and random.random() < 0.52:
                    continue
                if y > 0.92 and random.random() < 0.38:
                    continue

            px, py = self.heart_to_canvas(x, y)
            jitter = 1.2 if kind == "core" else 1.6
            return px + random.gauss(0.0, jitter), py + random.gauss(0.0, jitter)

        t = random.uniform(0, math.tau)
        px, py = self.heart_point(t, scale=random.uniform(0.4, 1.0))
        return self.center_x + px * 10.4, self.center_y + 18.0 + py * 10.4

    def make_heart_particle(self, kind: str) -> Particle:
        x, y = self.sample_heart_xy(kind)

        if kind == "glow":
            base_size = random.uniform(0.9, 1.8)
            color = self.hsv_to_hex(
                random.uniform(0.94, 0.99),
                random.uniform(0.5, 0.78),
                random.uniform(0.32, 0.52),
            )
            depth = random.uniform(0.4, 0.7)
            wobble = random.uniform(0.9, 1.9)
        elif kind == "core":
            base_size = random.uniform(1.0, 1.8)
            color = self.hsv_to_hex(
                random.uniform(0.96, 1.0),
                random.uniform(0.45, 0.75),
                random.uniform(0.92, 1.0),
            )
            depth = random.uniform(0.9, 1.18)
            wobble = random.uniform(0.18, 0.48)
        else:
            base_size = random.uniform(0.8, 1.35)
            color = self.hsv_to_hex(
                random.uniform(0.94, 0.99),
                random.uniform(0.7, 0.96),
                random.uniform(0.68, 0.95),
            )
            depth = random.uniform(0.74, 1.02)
            wobble = random.uniform(0.25, 0.7)

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
        radial = random.betavariate(1.25, 2.4)
        radius_x = radial * random.uniform(140, 250)
        radius_y = radial * random.uniform(6, 14)
        x = self.center_x + math.cos(angle) * radius_x
        y = self.center_y + 230 + math.sin(angle) * radius_y
        brightness = 0.66 + (1.0 - radial) * 0.2
        color = self.hsv_to_hex(
            random.uniform(0.55, 0.59),
            random.uniform(0.62, 0.82),
            min(1.0, brightness + random.uniform(-0.04, 0.04)),
        )
        particle = Particle(
            home_x=x,
            home_y=y,
            x=x,
            y=y,
            base_size=random.uniform(0.65, 1.35),
            color=color,
            kind="ground",
            depth=random.uniform(0.4, 0.85),
            wobble=random.uniform(1.0, 2.8),
            speed=random.uniform(0.18, 0.5),
            phase=random.uniform(0, math.tau),
            twinkle=random.uniform(0.25, 0.75),
        )
        particle.size = particle.base_size
        return particle

    def make_spark_particle(self) -> Particle:
        t = random.uniform(0, math.tau)
        px, py = self.heart_point(t, scale=random.uniform(1.0, 1.08))
        x = self.center_x + px * 10.9
        y = self.center_y + 8.0 + py * 10.9
        color = self.hsv_to_hex(random.uniform(0.09, 0.14), random.uniform(0.18, 0.4), 1.0)
        particle = Particle(
            home_x=x,
            home_y=y,
            x=x,
            y=y,
            base_size=random.uniform(0.55, 1.2),
            color=color,
            kind="spark",
            depth=random.uniform(0.8, 1.2),
            wobble=random.uniform(1.5, 3.2),
            speed=random.uniform(1.0, 2.2),
            phase=random.uniform(0, math.tau),
            twinkle=random.uniform(1.6, 3.4),
        )
        particle.size = particle.base_size
        return particle

    def _draw_background(self) -> None:
        self.canvas.delete("background")
        palette = ["#04060b", "#06101a", "#081424", "#09111a", "#060b14", "#04060b"]
        band_height = self.height / len(palette)
        for index, color in enumerate(palette):
            y0 = band_height * index
            y1 = band_height * (index + 1)
            self.canvas.create_rectangle(0, y0, self.width, y1, fill=color, outline="", tags="background")

        self.canvas.create_line(
            180,
            self.center_y + 232,
            self.width - 180,
            self.center_y + 232,
            fill="#0a1625",
            width=1,
            tags="background",
        )
        self.canvas.create_line(
            230,
            self.center_y + 240,
            self.width - 230,
            self.center_y + 240,
            fill="#07101a",
            width=1,
            tags="background",
        )

        self.star_items.clear()
        for _ in range(48):
            x = random.uniform(20, self.width - 20)
            y = random.uniform(20, self.height - 270)
            size = random.choice((0.8, 1.0, 1.0, 1.2, 1.4))
            color = random.choice(("#dce9ff", "#ffeef5", "#c9e7ff", "#fff3d7"))
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

        self.highlight_items = []
        for color in ("#fff4f8", "#ffe8ef", "#ffd8e3", "#fff9fb", "#ffeef3"):
            item_id = self.canvas.create_oval(0, 0, 0, 0, fill=color, outline="")
            self.highlight_items.append(item_id)
            self.canvas.tag_raise(item_id)

    def _build_scene(self) -> None:
        self.canvas.delete("all")
        self._draw_background()

        self.heart_particles = []
        self.ground_particles = []
        self.spark_particles = []

        for _ in range(250):
            self.heart_particles.append(self.make_heart_particle("glow"))
        for _ in range(2550):
            self.heart_particles.append(self.make_heart_particle("fill"))
        for _ in range(360):
            self.heart_particles.append(self.make_heart_particle("core"))
        for _ in range(320):
            self.ground_particles.append(self.make_ground_particle())
        for _ in range(70):
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
        shimmer = 0.6 + 1.2 * self.heartbeat
        pulse_offset = (scale - 1.0) * 28.0
        specs = (
            (-23.0, -138.0, 5.6),
            (-15.0, -129.0, 3.6),
            (-9.0, -122.0, 2.3),
            (-29.0, -127.0, 1.8),
            (-18.0, -116.0, 1.3),
        )
        for index, (item_id, (offset_x, offset_y, base_size)) in enumerate(zip(self.highlight_items, specs)):
            twinkle = 0.92 + 0.14 * math.sin(self.time * 2.6 + index * 0.85)
            radius = base_size * twinkle + shimmer
            x = self.center_x + offset_x + pulse_offset * 0.18
            y = self.center_y + offset_y - pulse_offset
            self.canvas.coords(item_id, x - radius, y - radius, x + radius, y + radius)

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
            drift_y = math.cos(self.time * particle.speed * 0.7 + particle.phase) * (particle.wobble * 0.08)
            pulse = 1.0 + 0.05 * self.heartbeat * particle.depth
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

            if particle.kind == "glow":
                flutter_scale = 1.0
                spring = 0.16
                damping = 0.75
                lift = -1.0 * self.heartbeat * particle.depth
                size_strength = 0.12
            elif particle.kind == "core":
                flutter_scale = 0.25
                spring = 0.21
                damping = 0.78
                lift = -0.65 * self.heartbeat * particle.depth
                size_strength = 0.09
            else:
                flutter_scale = 0.38
                spring = 0.19
                damping = 0.77
                lift = -0.72 * self.heartbeat * particle.depth
                size_strength = 0.08

            flutter_x = math.cos(self.time * particle.speed + particle.phase) * particle.wobble * flutter_scale
            flutter_y = math.sin(self.time * (particle.speed * 1.15) + particle.phase) * particle.wobble * flutter_scale

            particle.vx = (target_x + flutter_x - particle.x) * spring + particle.vx * damping
            particle.vy = (target_y + flutter_y + lift - particle.y) * spring + particle.vy * damping
            particle.x += particle.vx
            particle.y += particle.vy

            size_pulse = 1.0 + size_strength * self.heartbeat * particle.depth
            twinkle = 0.94 + 0.06 * math.sin(self.time * particle.twinkle + particle.phase)
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
            orbit = 6 + particle.wobble + 6 * self.heartbeat
            particle.x = particle.home_x + math.cos(angle) * orbit
            particle.y = particle.home_y + math.sin(angle * 1.2) * (orbit * 0.4) - 7 * self.heartbeat
            particle.size = particle.base_size * (
                0.82 + 0.35 * abs(math.sin(self.time * particle.twinkle + particle.phase))
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

